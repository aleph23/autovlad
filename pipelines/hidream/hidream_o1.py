from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from tqdm.rich import tqdm

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from pipelines.hidream.scheduler_flashfloweuler import FlashFlowMatchEulerDiscreteScheduler
from pipelines.hidream.scheduler_flowunipc import FlowUniPCMultistepScheduler
from pipelines.hidream.hidream_o1_utils import (
    CONDITION_IMAGE_SIZE,
    DEFAULT_TIMESTEPS,
    NOISE_SCALE,
    PATCH_SIZE,
    TIMESTEP_TOKEN_NUM,
    T_EPS,
    _calculate_dimensions,
    _ensure_special_tokens,
    _image_to_patch_tensor,
    _patches_to_np,
    _pil_to_normalized_tensor,
    _resize_pilimage,
    build_t2i_text_sample,
    get_rope_index_fix_point,
)

use_flash_attn = False
try:
    import flash_attn # pylint: disable=unused-import
    use_flash_attn = True
except ImportError:
    pass


@dataclass
class HiDreamO1PipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]


class HiDreamO1Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["latents"]
    vae_scale_factor = 1

    def __init__(
        self,
        transformer,
        processor,
        tokenizer,
        scheduler,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, processor=processor, scheduler=scheduler, tokenizer=tokenizer)
        _ensure_special_tokens(self.tokenizer)

    def _build_scheduler(
        self,
        num_inference_steps: int,
        shift: float,
        device: torch.device,
    ):
        if num_inference_steps <= 28:
            self.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            timesteps_list = DEFAULT_TIMESTEPS if num_inference_steps == 28 else None
        else:
            self.scheduler = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
            timesteps_list = None
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if timesteps_list is not None:
            self.scheduler.timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
            sigmas = [t.item() / 1000.0 for t in self.scheduler.timesteps]
            sigmas.append(0.0)
            self.scheduler.sigmas = torch.tensor(sigmas, device=device)

    def _prepare_reference_paths(
        self,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]],
        ref_images: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]],
    ) -> List[PIL.Image.Image]:
        refs: List[PIL.Image.Image] = []
        if image is not None:
            if isinstance(image, list):
                refs.extend(image)
            else:
                refs.append(image)
        if ref_images is not None:
            if isinstance(ref_images, list):
                refs.extend(ref_images)
            else:
                refs.append(ref_images)
        return refs

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        ref_images: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        height: int = 1440,
        width: int = 2560,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 3.0,
        timesteps_list: Optional[List[int]] = None,
        scheduler: Optional[Union[FlashFlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler]] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        noise_scale_start: float = NOISE_SCALE,
        noise_scale_end: float = NOISE_SCALE,
        noise_clip_std: float = 0.0,
        keep_original_aspect: bool = True,
        callback_on_step_end: Optional[Callable[[DiffusionPipeline, int, int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[HiDreamO1PipelineOutput, tuple]:
        model = self.transformer
        processor = self.processor
        tokenizer = self.tokenizer
        model_config = model.config

        if isinstance(prompt, list):
            prompt = prompt[0] if len(prompt) > 0 else ""
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0] if len(negative_prompt) > 0 else ""
        if num_inference_steps <= 28:
            guidance_scale = 1.0

        device = self._execution_device
        try:
            dtype = next(model.parameters()).dtype
        except (StopIteration, AttributeError, TypeError):
            dtype = torch.bfloat16

        if callback_on_step_end_tensor_inputs is not None:
            invalid_inputs = [name for name in callback_on_step_end_tensor_inputs if name not in self._callback_tensor_inputs]
            if invalid_inputs:
                raise ValueError(
                    f"callback_on_step_end_tensor_inputs has to be in {self._callback_tensor_inputs}, but found {invalid_inputs}"
                )

        refs = [img.convert("RGB") for img in self._prepare_reference_paths(image, ref_images)]
        preresized_ref_pil = None

        if keep_original_aspect and len(refs) >= 1:
            preresized_ref_pil = _resize_pilimage(refs[0], 2048, PATCH_SIZE)
            width, height = preresized_ref_pil.size
        else:
            width = max(PATCH_SIZE, int(round(width / PATCH_SIZE)) * PATCH_SIZE)
            height = max(PATCH_SIZE, int(round(height / PATCH_SIZE)) * PATCH_SIZE)

        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE

        if len(refs) == 0:
            cond_sample = build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config)
            uncond_sample = None
            if guidance_scale > 1.0:
                uncond_sample = build_t2i_text_sample(negative_prompt or " ", height, width, tokenizer, processor, model_config)

            def to_device(sample):
                return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}

            cond_sample = to_device(cond_sample)
            if uncond_sample is not None:
                uncond_sample = to_device(uncond_sample)

            ref_patches = None
            tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
            samples = [cond_sample]
            if uncond_sample is not None:
                samples.append(uncond_sample)
        else:
            image_token_id = model_config.image_token_id
            video_token_id = model_config.video_token_id
            vision_start_token_id = model_config.vision_start_token_id
            spatial_merge_size = model_config.vision_config.spatial_merge_size

            ref_pils = [preresized_ref_pil] if preresized_ref_pil is not None else refs
            k_refs = len(ref_pils)

            max_size = max(height, width)
            if k_refs == 2:
                max_size = max_size * 48 // 64
            elif k_refs <= 4:
                max_size = max_size // 2
            elif k_refs <= 8:
                max_size = max_size * 24 // 64
            elif k_refs > 8:
                max_size = max_size // 4

            ref_pils_resized, ref_patches_list = [], []
            for pil in ref_pils:
                pil_r = pil if (preresized_ref_pil is not None and pil is preresized_ref_pil) else _resize_pilimage(pil, max_size, PATCH_SIZE)
                ref_pils_resized.append(pil_r)
                x = _pil_to_normalized_tensor(pil_r).unsqueeze(0)
                x = _image_to_patch_tensor(x, patch_size=PATCH_SIZE).squeeze(0)
                ref_patches_list.append(x)

            ref_image_lens = [img.shape[0] for img in ref_patches_list]
            total_ref_len = sum(ref_image_lens)
            ref_patches = torch.cat(ref_patches_list, dim=0).unsqueeze(0).to(device, dtype)

            tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

            cond_img_size = CONDITION_IMAGE_SIZE
            if k_refs > 4 and k_refs <= 8:
                cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
            elif k_refs > 8:
                cond_img_size = CONDITION_IMAGE_SIZE // 2

            ref_pils_vlm = []
            for pil_r in ref_pils_resized:
                cond_w, cond_h = _calculate_dimensions(cond_img_size, pil_r.width / pil_r.height)
                ref_pils_vlm.append(pil_r.resize((cond_w, cond_h), resample=PIL.Image.Resampling.LANCZOS))

            image_grid_thw_tgt = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
            image_grid_thw_ref = torch.zeros((k_refs, 3), dtype=torch.int64)
            for i, pil_r in enumerate(ref_pils_resized):
                rw, rh = pil_r.size
                image_grid_thw_ref[i] = torch.tensor([1, rh // PATCH_SIZE, rw // PATCH_SIZE], dtype=torch.int64)

            samples = []
            captions = [prompt]
            if guidance_scale > 1.0:
                captions.append(negative_prompt or " ")

            for caption in captions:
                boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
                tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

                content = [{"type": "image"} for _ in range(k_refs)]
                content.append({"type": "text", "text": caption})
                messages = [{"role": "user", "content": content}]

                template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")

                input_ids_2 = tokenizer.encode(boi_token + tms_token * TIMESTEP_TOKEN_NUM, return_tensors="pt", add_special_tokens=False)
                input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

                igthw_cond = proc.image_grid_thw.clone()
                for i in range(k_refs):
                    igthw_cond[i, 1] //= spatial_merge_size
                    igthw_cond[i, 2] //= spatial_merge_size
                igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

                vision_tokens_list = []
                vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
                vt_tgt[0, 0] = vision_start_token_id
                vision_tokens_list.append(vt_tgt)
                for ref_len in ref_image_lens:
                    vt_ref = torch.full((1, ref_len), image_token_id, dtype=input_ids.dtype)
                    vt_ref[0, 0] = vision_start_token_id
                    vision_tokens_list.append(vt_ref)
                vision_tokens = torch.cat(vision_tokens_list, dim=1)
                input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

                position_ids, _ = get_rope_index_fix_point(
                    1,
                    image_token_id,
                    video_token_id,
                    vision_start_token_id,
                    input_ids=input_ids_pad,
                    image_grid_thw=igthw_all,
                    video_grid_thw=None,
                    attention_mask=None,
                    skip_vision_start_token=[0] * k_refs + [1] + [1] * k_refs,
                )

                txt_seq_len = input_ids.shape[-1]
                all_seq_len = position_ids.shape[-1]

                token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
                bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
                end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
                token_types_raw[0, bgn:end] = 1
                token_types_raw[0, end : end + total_ref_len] = 2
                token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

                vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
                token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

                samples.append(
                    {
                        "input_ids": input_ids_pad.to(device),
                        "position_ids": position_ids.to(device),
                        "token_types": token_types_bin.to(device),
                        "vinput_mask": vinput_mask.to(device),
                        "pixel_values": proc.pixel_values.to(device, dtype),
                        "image_grid_thw": proc.image_grid_thw.to(device),
                    }
                )

        if generator is None:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed + 1)

        noise = noise_scale_start * randn_tensor((1, 3, height, width), generator=generator, device=device, dtype=dtype)
        z = _image_to_patch_tensor(noise, patch_size=PATCH_SIZE)


        if scheduler is None:
            self._build_scheduler(
                num_inference_steps=num_inference_steps,
                shift=shift,
                device=device,
            )

        num_steps = len(self.scheduler.timesteps)
        if num_steps > 1:
            noise_scale_schedule = [noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1) for i in range(num_steps)]
        else:
            noise_scale_schedule = [noise_scale_start]

        def forward_once(sample: Dict[str, torch.Tensor], z_in: torch.Tensor, t_pixeldit: torch.Tensor):
            kwargs: Dict[str, Any] = {
                "input_ids": sample["input_ids"],
                "position_ids": sample["position_ids"],
                "vinputs": z_in,
                "timestep": t_pixeldit.reshape(-1).to(device),
                "token_types": sample["token_types"],
            }
            if use_flash_attn is not None:
                kwargs["use_flash_attn"] = use_flash_attn
            if "pixel_values" in sample:
                kwargs["pixel_values"] = sample["pixel_values"]
            if "image_grid_thw" in sample:
                kwargs["image_grid_thw"] = sample["image_grid_thw"]

            outputs = model(**kwargs)
            x_pred = outputs.x_pred
            if ref_patches is None:
                return x_pred[0, sample["vinput_mask"][0]].unsqueeze(0)
            return x_pred[0, sample["vinput_mask"][0]][:tgt_image_len].unsqueeze(0)

        preview_x0 = None
        for step_idx, step_t in enumerate(tqdm(self.scheduler.timesteps, desc="Processing", unit="it")):
            t_pixeldit = 1.0 - step_t.float() / 1000.0
            sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

            if ref_patches is None:
                x_pred_cond = forward_once(samples[0], z.clone(), t_pixeldit)
                v_cond = (x_pred_cond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma
                if len(samples) > 1:
                    x_pred_uncond = forward_once(samples[1], z.clone(), t_pixeldit)
                    v_uncond = (x_pred_uncond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_guided = v_cond
                preview_x0 = x_pred_cond
            else:
                vinputs = torch.cat([z, ref_patches], dim=1)
                x_vis_list = [forward_once(sample, vinputs, t_pixeldit) for sample in samples]
                x_vis_stacked = torch.cat(x_vis_list, dim=0)

                z_rep = z.expand(len(samples), -1, -1)
                v_pred = (x_vis_stacked.to(dtype=torch.float32) - z_rep.to(dtype=torch.float32)) / sigma
                v_cond = v_pred[0:1]
                if len(samples) > 1:
                    v_uncond = v_pred[1:]
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_guided = v_cond
                preview_x0 = x_vis_list[0]

            model_output = -v_guided
            if num_inference_steps <= 28:
                z = self.scheduler.step(
                    model_output.float(),
                    step_t.to(dtype=torch.float32),
                    z.float(),
                    s_noise=noise_scale_schedule[step_idx],
                    noise_clip_std=noise_clip_std,
                    return_dict=False,
                )[0].to(dtype)
            else:
                z = self.scheduler.step(model_output.float(), step_t.to(dtype=torch.float32), z.float(), return_dict=False)[0].to(dtype)

            if callback_on_step_end is not None:
                pil_image = PIL.Image.fromarray(_patches_to_np(preview_x0, h_patches, w_patches, invert=False))
                callback_on_step_end(self, step_idx, int(step_t.item()), {"image": pil_image})

        if output_type == "pil":
            output_images = [PIL.Image.fromarray(_patches_to_np(preview_x0, h_patches, w_patches, invert=False, rescale=False))]
        elif output_type == "np":
            output_images = [_patches_to_np(preview_x0, h_patches, w_patches, invert=True, rescale=True)]
        else:
            raise ValueError(f"Unsupported output_type={output_type!r}; supported values are 'pil' and 'np'")

        if not return_dict:
            return (output_images,)
        return HiDreamO1PipelineOutput(images=output_images)


class HiDreamO1ImagePipeline(HiDreamO1Pipeline):
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        height: int = 1440,
        width: int = 2560,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 3.0,
        timesteps_list: Optional[List[int]] = None,
        scheduler: Optional[Union[FlashFlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler]] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        noise_scale_start: float = NOISE_SCALE,
        noise_scale_end: float = NOISE_SCALE,
        noise_clip_std: float = 0.0,
        keep_original_aspect: bool = True,
        use_flash_attn: Optional[bool] = None,
        callback: Optional[Callable[[int, int, Callable[[], PIL.Image.Image]], None]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[HiDreamO1PipelineOutput, tuple]:
        # image is list, first entry should go to image and remaining to ref_images
        ref_images = []
        if isinstance(image, list):
            ref_images = image[1:]
            image = image[0] if len(image) > 0 else None
        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            ref_images=ref_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            timesteps_list=timesteps_list,
            scheduler=scheduler,
            generator=generator,
            seed=seed,
            noise_scale_start=noise_scale_start,
            noise_scale_end=noise_scale_end,
            noise_clip_std=noise_clip_std,
            keep_original_aspect=keep_original_aspect,
            callback=callback,
            output_type=output_type,
            return_dict=return_dict,
            **kwargs,
        )

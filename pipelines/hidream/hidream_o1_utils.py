import math
from typing import Optional

import numpy as np
import PIL.Image
import torch


TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 8.0
T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
PATCH_SIZE = 32

DEFAULT_TIMESTEPS = [
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
]

PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]


def _ensure_special_tokens(tokenizer):
    if not hasattr(tokenizer, "boi_token"):
        tokenizer.boi_token = "<|boi_token|>"
    if not hasattr(tokenizer, "bor_token"):
        tokenizer.bor_token = "<|bor_token|>"
    if not hasattr(tokenizer, "eor_token"):
        tokenizer.eor_token = "<|eor_token|>"
    if not hasattr(tokenizer, "bot_token"):
        tokenizer.bot_token = "<|bot_token|>"
    if not hasattr(tokenizer, "tms_token"):
        tokenizer.tms_token = "<|tms_token|>"


def _find_closest_resolution(width: int, height: int):
    img_ratio = width / height
    best_res = PREDEFINED_RESOLUTIONS[0]
    min_diff = float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        diff = abs((w / h) - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
    return best_res


def _resize_pilimage(
    pil_image: PIL.Image.Image,
    image_size: int,
    patch_size: int = 16,
    resampler: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=PIL.Image.Resampling.BOX)

    width, height = pil_image.width, pil_image.height
    max_area = image_size * image_size
    scale = math.sqrt(max_area / (width * height))

    m = patch_size
    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    new_size = new_sizes[-1]
    for candidate in new_sizes:
        if candidate[0] * candidate[1] <= max_area:
            new_size = candidate
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=resampler)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=resampler)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

    return pil_image


def _calculate_dimensions(max_size: int, ratio: float):
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / 32) * 32
    height = int(height / 32) * 32
    return width, height


def _image_to_patch_tensor(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    b, c, h, w = x.shape
    h_patch = h // patch_size
    w_patch = w // patch_size
    x = x.reshape(b, c, h_patch, patch_size, w_patch, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, h_patch * w_patch, c * patch_size * patch_size)


def _patch_tensor_to_image(x: torch.Tensor, h_patches: int, w_patches: int, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    b = x.shape[0]
    c = x.shape[-1] // (patch_size * patch_size)
    x = x.reshape(b, h_patches, w_patches, c, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h_patches * patch_size, w_patches * patch_size)


def _pil_to_normalized_tensor(image: PIL.Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    return x * 2.0 - 1.0


def _patches_to_np(
    z: torch.Tensor,
    h_patches: int,
    w_patches: int,
    invert: bool = True,
    rescale: bool = False,
) -> np.ndarray:
    z = z.float()
    if rescale:
        clip_ratio = (torch.abs(z) > 1.0).float().mean().item() # hidream-o1 often has outlines and clipping
        if clip_ratio > 0.10: # Balanced quantiles and headroom: compress outliers while preserving contrast.
            lo_q, hi_q, headroom = 0.03, 0.97, 1.33
        elif clip_ratio > 0.03:
            lo_q, hi_q, headroom = 0.02, 0.98, 1.22
        else:
            lo_q, hi_q, headroom = 0.01, 0.99, 1.15
        z_flat = z.reshape(-1)
        q_lo = torch.quantile(z_flat, lo_q)
        q_hi = torch.quantile(z_flat, hi_q)
        center = (q_hi + q_lo) * 0.5
        half_range = (q_hi - q_lo) * 0.5
        if half_range > 0:
            z = (z - center) / torch.clamp(half_range * headroom, min=0.35) # Balanced robust remap: moderate compression to handle outliers without losing contrast.
    z = z.clamp(-1.0, 1.0)
    image = (1.0 - z) / 2.0 if invert else (z + 1.0) / 2.0
    image = _patch_tensor_to_image(image, h_patches=h_patches, w_patches=w_patches, patch_size=PATCH_SIZE)
    np_image = image[0].cpu().numpy().transpose(1, 2, 0)
    np_image = np.round(255.0 * np_image).astype(np.uint8)
    return np_image


def get_rope_index_fix_point(
    spatial_merge_size,
    image_token_id,
    video_token_id,
    vision_start_token_id,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    skip_vision_start_token=None,
    fix_point=4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
    return position_ids, mrope_position_deltas


def build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + boi_token + tms_token * TIMESTEP_TOKEN_NUM
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn : bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

    vinput_mask = token_types == 1
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        "input_ids": input_ids_pad,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }


__all__ = [
    "CONDITION_IMAGE_SIZE",
    "DEFAULT_TIMESTEPS",
    "NOISE_SCALE",
    "PATCH_SIZE",
    "PREDEFINED_RESOLUTIONS",
    "TIMESTEP_TOKEN_NUM",
    "T_EPS",
    "_calculate_dimensions",
    "_ensure_special_tokens",
    "_find_closest_resolution",
    "_image_to_patch_tensor",
    "_patch_tensor_to_image",
    "_patches_to_np",
    "_pil_to_normalized_tensor",
    "_resize_pilimage",
    "build_t2i_text_sample",
    "get_rope_index_fix_point",
]

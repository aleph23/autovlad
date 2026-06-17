from pipelines.lumina_dimmo.lumina_dimoo import LuminaDiMOOPipeline
from modules.logger import log


class LuminaDiMOOTextPipeline(LuminaDiMOOPipeline):
    def __call__(self, prompt: list[str], height: int = 1024, width: int = 1024, num_inference_steps: int = 64, cfg_scale: float = 4.0, temperature: float = 1.0, painting_mode=None, painting_image=None, mask_h_ratio: float = 1.0, mask_w_ratio: float = 0.2, use_cache: bool = True, cache_ratio: float = 0.9, refresh_interval: int = 5, warmup_ratio: float = 0.3, **kwargs):
        if isinstance(prompt, list):
            prompt = prompt[0].strip()
        task = "text_to_image"
        log.debug(f'Base args: task={task} height={height} width={width} steps={num_inference_steps} cfg_scale={cfg_scale} temperature={temperature} painting_mode={painting_mode} mask_h_ratio={mask_h_ratio} mask_w_ratio={mask_w_ratio} use_cache={use_cache} cache_ratio={cache_ratio} refresh_interval={refresh_interval} warmup_ratio={warmup_ratio}')
        return super().__call__(
            prompt=prompt,
            task=task,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            temperature=temperature,
            painting_mode=painting_mode,
            painting_image=painting_image,
            mask_h_ratio=mask_h_ratio,
            mask_w_ratio=mask_w_ratio,
            use_cache=use_cache,
            cache_ratio=cache_ratio,
            refresh_interval=refresh_interval,
            warmup_ratio=warmup_ratio,
            **kwargs)


class LuminaDiMOOImagePipeline(LuminaDiMOOPipeline):
    def __call__(self, prompt: list[str], image, num_inference_steps: int = 64, temperature: float = 1.0, cfg_scale: float = 2.5, cfg_img: float = 4.0, **kwargs):
        edit_types = ['dense', 'canny_pred', 'control', 'subject', 'edit', 'ref_transfer', 'multi_view']
        if isinstance(prompt, list):
            prompt = prompt[0].strip()
        task = "image_to_image"
        edit_type = 'default'
        ref_image = None
        for et in edit_types:
            if prompt.startswith(et):
                edit_type = et
                break
        if isinstance(image, list):
            if len(image) > 1:
                ref_image = image[1]
            if len(image) > 0:
                image = image[0]
        log.debug(f'Base args: task={task} edit_type={edit_type} steps={num_inference_steps} cfg_scale={cfg_scale} cfg_img={cfg_img} temperature={temperature} image={image} ref_image={ref_image}')
        return super().__call__(
            prompt=prompt,
            task=task,
            image=image,
            ref_image=ref_image,
            edit_type=edit_type,
            num_inference_steps=num_inference_steps,
            temperature=temperature,
            cfg_scale=cfg_scale,
            cfg_img=cfg_img,
            **kwargs)

import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from diffusers import FluxPipeline

MODEL_DIR = "/data_hdd/lzc/models/flux-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


@dataclass
class MaskStep:
    timestep: int
    iou_to_final: float
    area_ratio: float


def load_pipe(model_dir: str, device: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.to(device)
    return pipe


def _encode_prompt(pipe: FluxPipeline, prompt: str, device: str, guidance_scale: float):
    signature = inspect.signature(pipe.encode_prompt)
    kwargs = {
        "prompt": prompt,
        "device": device,
        "do_classifier_free_guidance": guidance_scale > 1.0,
    }
    if "num_images_per_prompt" in signature.parameters:
        kwargs["num_images_per_prompt"] = 1
    if "negative_prompt" in signature.parameters:
        kwargs["negative_prompt"] = None
    return pipe.encode_prompt(**kwargs)


def _prepare_latents(pipe: FluxPipeline, batch_size: int, height: int, width: int, device: str):
    signature = inspect.signature(pipe.prepare_latents)
    kwargs = {
        "batch_size": batch_size,
        "num_channels_latents": pipe.transformer.config.in_channels,
        "height": height,
        "width": width,
        "dtype": pipe.transformer.dtype,
        "device": device,
    }
    if "generator" in signature.parameters:
        kwargs["generator"] = None
    return pipe.prepare_latents(**kwargs)


def _get_timesteps(pipe: FluxPipeline, num_inference_steps: int, device: str):
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    return pipe.scheduler.timesteps


def _call_transformer(transformer, latents, t, prompt_embeds, pooled_prompt_embeds, guidance):
    signature = inspect.signature(transformer.forward)
    kwargs = {}
    if "encoder_hidden_states" in signature.parameters:
        kwargs["encoder_hidden_states"] = prompt_embeds
    if "pooled_projections" in signature.parameters:
        kwargs["pooled_projections"] = pooled_prompt_embeds
    if "timestep" in signature.parameters:
        kwargs["timestep"] = t
    if "guidance" in signature.parameters:
        kwargs["guidance"] = guidance
    output = transformer(latents, **kwargs)
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, tuple):
        return output[0]
    return output


def _delta_map(delta: torch.Tensor) -> torch.Tensor:
    magnitude = delta.float().abs().mean(dim=1)
    magnitude = magnitude / (magnitude.amax(dim=(-1, -2), keepdim=True) + 1e-8)
    return magnitude


def _smooth_mask(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    kernel = kernel / kernel.sum()
    filtered = torch.nn.functional.conv2d(mask.unsqueeze(1), kernel, padding=pad)
    return filtered.squeeze(1)


def _adaptive_threshold(mask: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    flat = mask.flatten(1)
    k = max(1, int(flat.shape[1] * keep_ratio))
    topk = torch.topk(flat, k=k, dim=1).values[:, -1:]
    return (mask >= topk.view(-1, 1, 1)).float()


def _mask_from_delta(delta: torch.Tensor, keep_ratio: float, smooth: int) -> torch.Tensor:
    mask = _delta_map(delta)
    mask = _smooth_mask(mask, kernel_size=smooth)
    return _adaptive_threshold(mask, keep_ratio)


def _iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    intersection = (mask_a * mask_b).sum().item()
    union = (mask_a + mask_b).clamp(max=1.0).sum().item() + 1e-8
    return intersection / union


def track_mask_convergence(
    prompt: str,
    model_dir: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: str,
    stage_ratio: float,
    keep_ratio: float,
    smooth: int,
) -> List[MaskStep]:
    pipe = load_pipe(model_dir, device)
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(pipe, prompt, device, guidance_scale)
    timesteps = _get_timesteps(pipe, num_inference_steps, device)
    latents = _prepare_latents(pipe, 1, height, width, device)

    stage_end = int(len(timesteps) * stage_ratio)
    stage_end = max(stage_end, 2)

    masks: List[torch.Tensor] = []
    for i, t in enumerate(timesteps[:stage_end]):
        dt = (timesteps[i + 1] - t).float() if i + 1 < len(timesteps) else -1.0
        velocity = _call_transformer(
            pipe.transformer, latents, t, prompt_embeds, pooled_prompt_embeds, guidance_scale
        )
        delta = dt * velocity
        masks.append(_mask_from_delta(delta, keep_ratio=keep_ratio, smooth=smooth))
        latents = latents + delta

    final_mask = masks[-1]
    results: List[MaskStep] = []
    final_area = final_mask.mean().item() + 1e-8
    for idx, (t, mask) in enumerate(zip(timesteps[:stage_end], masks)):
        iou = _iou(mask, final_mask)
        area_ratio = mask.mean().item() / final_area
        results.append(MaskStep(timestep=int(t.item()), iou_to_final=iou, area_ratio=area_ratio))
    return results


def _format_table(results: Iterable[MaskStep]) -> str:
    header = f"{'timestep':>10} | {'iou':>8} | {'area_ratio':>10}"
    rows = [header, "-" * len(header)]
    for item in results:
        rows.append(f"{item.timestep:>10} | {item.iou_to_final:>8.4f} | {item.area_ratio:>10.4f}")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track mask convergence during stage-1.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage-ratio", type=float, default=0.6)
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--smooth", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = track_mask_convergence(
        prompt=args.prompt,
        model_dir=args.model_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        device=args.device,
        stage_ratio=args.stage_ratio,
        keep_ratio=args.keep_ratio,
        smooth=args.smooth,
    )
    print(_format_table(results))


if __name__ == "__main__":
    main()

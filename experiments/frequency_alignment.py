import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from diffusers import FluxPipeline

MODEL_DIR = "/data_hdd/lzc/models/flux-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


@dataclass
class StepEnergy:
    timestep: int
    low: float
    mid: float
    high: float


@dataclass
class BandConfig:
    low_radius: float
    mid_radius: float


DEFAULT_BANDS = BandConfig(low_radius=0.15, mid_radius=0.35)


def load_pipe(model_dir: str, device: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.to(device)
    return pipe


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


def _heun_update(pipe: FluxPipeline, latents, t, dt, prompt_embeds, pooled_prompt_embeds, guidance):
    v_t = _call_transformer(pipe.transformer, latents, t, prompt_embeds, pooled_prompt_embeds, guidance)
    x_pred = latents + dt * v_t
    v_pred = _call_transformer(pipe.transformer, x_pred, t, prompt_embeds, pooled_prompt_embeds, guidance)
    delta_first = dt * v_t
    delta_second = 0.5 * dt * (v_t + v_pred)
    delta_acc = delta_second - delta_first
    return delta_first, delta_second, delta_acc


def _fft_energy_map(delta: torch.Tensor) -> torch.Tensor:
    spatial = delta.float().mean(dim=1)
    fft_map = torch.fft.fft2(spatial)
    fft_shift = torch.fft.fftshift(fft_map)
    return (fft_shift.real ** 2 + fft_shift.imag ** 2)


def _radial_bands(energy_map: torch.Tensor, bands: BandConfig) -> Tuple[float, float, float]:
    _, height, width = energy_map.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=energy_map.device),
        torch.linspace(-1.0, 1.0, width, device=energy_map.device),
        indexing="ij",
    )
    radius = torch.sqrt(x ** 2 + y ** 2)
    low_mask = radius <= bands.low_radius
    mid_mask = (radius > bands.low_radius) & (radius <= bands.mid_radius)
    high_mask = radius > bands.mid_radius
    low = energy_map[:, low_mask].sum()
    mid = energy_map[:, mid_mask].sum()
    high = energy_map[:, high_mask].sum()
    total = low + mid + high + 1e-8
    return (low / total).item(), (mid / total).item(), (high / total).item()


def analyze_frequency_alignment(
    prompt: str,
    model_dir: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: str,
    bands: BandConfig,
) -> List[StepEnergy]:
    pipe = load_pipe(model_dir, device)
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(pipe, prompt, device, guidance_scale)
    timesteps = _get_timesteps(pipe, num_inference_steps, device)
    latents = _prepare_latents(pipe, 1, height, width, device)

    results: List[StepEnergy] = []
    for i, t in enumerate(timesteps[:-1]):
        dt = (timesteps[i + 1] - t).float()
        delta_first, delta_second, delta_acc = _heun_update(
            pipe, latents, t, dt, prompt_embeds, pooled_prompt_embeds, guidance_scale
        )
        energy_map = _fft_energy_map(delta_acc)
        low, mid, high = _radial_bands(energy_map, bands)
        results.append(StepEnergy(int(t.item()), low, mid, high))
        latents = latents + delta_second
    return results


def _format_table(results: Iterable[StepEnergy]) -> str:
    header = f"{'timestep':>10} | {'low':>8} | {'mid':>8} | {'high':>8}"
    rows = [header, "-" * len(header)]
    for item in results:
        rows.append(
            f"{item.timestep:>10} | {item.low:>8.4f} | {item.mid:>8.4f} | {item.high:>8.4f}"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequency-band alignment for Δx(a).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-radius", type=float, default=DEFAULT_BANDS.low_radius)
    parser.add_argument("--mid-radius", type=float, default=DEFAULT_BANDS.mid_radius)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bands = BandConfig(low_radius=args.low_radius, mid_radius=args.mid_radius)
    results = analyze_frequency_alignment(
        prompt=args.prompt,
        model_dir=args.model_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        device=args.device,
        bands=bands,
    )
    print(_format_table(results))


if __name__ == "__main__":
    main()

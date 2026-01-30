import torch
from diffusers import FluxPipeline

MODEL_DIR = "/data_hdd/lzc/models/flux-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"

pipe = FluxPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,   # 👈 强制不联网
)

pipe.enable_model_cpu_offload()

img = pipe(
    # "A photorealistic scene: on the left, a man in a bright red jacket; on the right, a woman in a bright yellow jacket. They are standing side by side, facing the camera, sharp focus, detailed textures.",
    # "A photorealistic scene: on the left, a man wearing a small blue baseball cap and round black glasses; on the right, a woman wearing a large red beret and square white glasses. Facing the camera, sharp focus.",
    # "A photorealistic studio scene:on the left, a man holding a large green sign with the word “LEFT” clearly visible;on the right, a woman holding a large red sign with the word “RIGHT” clearly visible.Both are standing upright, facing the camera, sharp focus, studio lighting, high detail.",
    # "A photorealistic studio scene: two people facing each other. On the left, a man wearing a small blue cap with a large white feather clearly visible. On the right, a woman wearing a large red beret with a large golden bell clearly visible. The man is pointing at the woman’s hat, sharp focus, ultra-detailed textures.",
    # "A photorealistic studio scene:on the left, a man in a bright red jacket holding a large green sign with the word “LEFT” clearly visible;on the right, a woman in a bright yellow jacket holding a large red sign with the word “RIGHT” clearly visible.Both are standing upright, facing the camera, sharp focus, studio lighting, high detail.",
    # "A photorealistic studio scene: on the left, a man in a bright red jacket holding a large green sign with the word \"LEFT\" clearly visible, wearing a black wristwatch on his left wrist; on the right, a woman in a bright yellow jacket holding a large red sign with the word \"RIGHT\" clearly visible, wearing a silver bracelet on her right wrist. Both are standing upright, facing the camera, sharp focus, studio lighting, high detail.",
    # "A photorealistic studio scene: on the left, a man in a bright red jacket holding a large green sign with the word \"LEFT\" clearly visible, slightly raising his left arm and smiling; on the right, a woman in a bright yellow jacket holding a large red sign with the word \"RIGHT\" clearly visible, slightly tilting her head to the right while standing confidently. Both are facing the camera, sharp focus, studio lighting, high detail.",
    # "A photorealistic studio scene: on the left, a man wearing a small blue baseball cap and round black glasses, holding a large green sign with the word \"LEFT\" clearly visible; on the right, a woman wearing a large red beret and square white glasses, holding a large red sign with the word \"RIGHT\" clearly visible. Both are standing upright, facing the camera, sharp focus, studio lighting, high detail.",
    # "A photorealistic scene: on the left, a man in a bright red jacket wearing a small blue baseball cap and round black glasses, holding a large green sign with the word \"LEFT\" clearly visible; on the right, a woman in a bright yellow jacket wearing a large red beret and square white glasses, holding a large red sign with the word \"RIGHT\" clearly visible. Both are standing upright, facing the camera, sharp focus,high detail.",
    "a man wearing a red hat and blue tracksuit is standing in front of a green sports car",
    num_inference_steps=30,
    guidance_scale=3.5,
    height=768, width=768,
).images[0]

img.save("flux_dev_test_1.png")
print("OK")

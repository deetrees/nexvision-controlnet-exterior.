import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="What changes to make to the home exterior"),
        num_inference_steps: int = Input(default=25),
        guidance_scale: float = Input(default=7.5),
    ) -> Path:
        input_image = Image.open(image).convert("RGB").resize((768, 768))
        result = self.pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)

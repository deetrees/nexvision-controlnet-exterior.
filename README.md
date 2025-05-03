# NexVision ControlNet Exterior Editor

This model uses ControlNet (Depth) and Stable Diffusion XL to enhance or modify exterior home images based on natural language prompts.

## Example Prompts:
- Add a white fence to the right side
- Replace the roof with black shingles
- Add bushes under the front windows

## Inputs
- `image`: A photo of a home exterior
- `prompt`: A description of what you want to change
- `num_inference_steps`: Number of steps for generation (default: 25)
- `guidance_scale`: How closely to follow the prompt (default: 7.5)

## Output
- A realistic, modified image of the home

Built by NexVision, powered by Cog and Replicate.

"""
Text2Image.py - Text-to-Image Generation with HuggingFace Diffusers (AmusedPipeline)

This script demonstrates how to generate an image from a text prompt using a pre-trained diffusion model pipeline from the HuggingFace Diffusers library. 
It is designed for production-quality, end-to-end text-to-image synthesis, and can be adapted for other prompts, models, or output formats.

Overview:
---------
- Loads a pre-trained text-to-image diffusion model ("amused/amused-256") using the AmusedPipeline from HuggingFace Diffusers.
- Configures the model to use mixed precision (fp16 for most weights, fp32 for VQ-VAE) for optimal performance and stability.
- Generates an image from a user-specified text prompt.
- Saves the generated image to disk as a PNG file.

Key Components:
---------------
1. Model Selection and Loading:
	- The "amused/amused-256" model is a state-of-the-art text-to-image diffusion model trained to generate 256x256 pixel images from natural language prompts.
	- The model is loaded with the `variant="fp16"` argument to use half-precision weights, reducing memory usage and improving speed on compatible hardware (e.g., modern GPUs).
	- The `torch_dtype=torch.float16` argument ensures the model weights are loaded in half-precision.

2. Precision Handling:
	- The pipeline's VQ-VAE (Vector Quantized Variational Autoencoder) component is explicitly converted to 32-bit floating point (`float32`) for numerical stability during image decoding. This is sometimes required for correct operation, especially if the VQ-VAE does not support half-precision.

3. Image Generation:
	- The pipeline is called with a text prompt ("A fantasy landscape, trending on artstation") to generate an image that matches the description.
	- The output is a list of images; the first image is selected for saving.

4. Output:
	- The generated image is saved as 'text2image_256.png' in the current working directory.
	- The output file can be opened with any standard image viewer.

Best Practices:
---------------
- For production use, validate the input prompt and handle exceptions during model loading and inference.
- To use a different model or generate higher-resolution images, change the model name or pipeline parameters as needed.
- For batch processing, loop over a list of prompts and save each generated image with a unique filename.
- For web or interactive applications, return the image object directly instead of saving to disk.

Dependencies:
-------------
- diffusers >= 0.14.0
- torch (PyTorch backend)
- PIL (for image saving, typically installed with diffusers)

References:
-----------
- HuggingFace Diffusers documentation: https://huggingface.co/docs/diffusers/index
- AmusedPipeline: https://huggingface.co/amused/amused-256
- VQ-VAE: https://arxiv.org/abs/1711.00937

"""
import torch
from diffusers import AmusedPipeline

#%%
pipe = AmusedPipeline.from_pretrained(
"amused/amused-256", variant="fp16", torch_dtype=torch.float16
)
pipe.vqvae.to(torch.float32)

image = pipe("A fantasy landscape, trending on artstation").images[0]
image.save('text2image_256.png')
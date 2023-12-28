
# Text to Image Generation with Diffusion Models

Random Noise -> Encoder(vae.py) -> Latent Space Z -> Time Scheduler + Latent Space + Feature Extractions (unet.py) -> Decoder (vae.py) -> Reconstruction (sampler.py, __init__.py)
Text Prompt (main.py) -> Clip Encoder (clip.py) -> Prompt Embedding (clip.py) -> Time Scheduler + Latent Space + Feature Extractions (unet.py) -> Decoder (vae.py) -> Reconstruction (sampler.py, __init__.py)

# Image to Image Generation with Diffusion Models

Base Image + Random Noise -> Encoder(vae.py) -> Latent Space Z -> Time Scheduler + Latent Space + Feature Extractions (unet.py) -> Decoder (vae.py) -> Reconstruction (sampler.py, __init__.py)
Text Prompt (main.py) -> Clip Encoder (clip.py) -> Prompt Embedding (clip.py) -> Time Scheduler + Latent Space + Feature Extractions (unet.py) -> Decoder (vae.py) -> Reconstruction (sampler.py, __init__.py)



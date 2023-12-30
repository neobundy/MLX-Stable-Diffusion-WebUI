# Copyright © 2023 Apple Inc.

from typing import Tuple
from PIL import Image
import numpy as np
import mlx.core as mx
import streamlit as st
from tqdm import tqdm

from .model_io import (
    load_unet,
    load_text_encoder,
    load_autoencoder,
    load_diffusion_config,
    load_tokenizer,
    _DEFAULT_MODEL,
)
from .sampler import SimpleEulerSampler, DDPMSampler

from utils import debug_print, normalize_tensor, tensor_head, visualize_tensor, inspect_tensor, get_time_embedding

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LATENTS_WIDTH = IMAGE_WIDTH // 8
LATENTS_HEIGHT = IMAGE_HEIGHT // 8


def _repeat(x, n, axis):
    # Make the expanded shape
    s = x.shape
    s.insert(axis + 1, n)

    # Expand
    x = mx.broadcast_to(mx.expand_dims(x, axis + 1), s)

    # Make the flattened shape
    s.pop(axis + 1)
    s[axis] *= n

    return x.reshape(s)


class StableDiffusion:
    def __init__(self, model: str = _DEFAULT_MODEL, float16: bool = False):
        self.dtype = mx.float16 if float16 else mx.float32
        self.diffusion_config = load_diffusion_config(model)
        self.unet = load_unet(model, float16)
        self.text_encoder = load_text_encoder(model, float16)
        self.autoencoder = load_autoencoder(model, float16)
        self.sampler = DDPMSampler(self.diffusion_config)
        self.tokenizer = load_tokenizer(model)

    def generate_latents(
            self,
            text: str,
            input_image: Image.Image = None,
            n_images: int = 1,
            num_steps: int = 50,
            cfg_weight: float = 7.5,
            negative_text: str = "",
            latent_size: Tuple[int] = (64, 64),
            seed=-1,
            image_strength=0.7,
    ):

        # Set the PRNG state

        mx.random.seed(int(seed))

        debug_print(f"Seed: {seed}")

        # Tokenize the text
        tokens = [self.tokenizer.tokenize(text)]
        if cfg_weight > 1:
            tokens += [self.tokenizer.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [0] * (N - len(t)) for t in tokens]
        tokens = mx.array(tokens)

        # Compute the features
        conditioning = self.text_encoder(tokens)

        # Repeat the conditioning for each of the generated images
        if n_images > 1:
            conditioning = _repeat(conditioning, n_images, axis=0)

        # Create the latent variables
        if input_image is not None:
            # number of values to print from the tensor
            num_values_to_inspect = 2

            # Convert the input image to an mlx array
            input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            input_image = mx.array(np.array(input_image))
            input_image = normalize_tensor(input_image, (0, 255), (-1, 1))
            input_image = mx.expand_dims(input_image, axis=0).astype(self.dtype)

            # Encode the input image to get the latent representation
            latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
            encoder_noise = mx.random.normal(shape=latents_shape, dtype=self.dtype)
            latents = self.autoencoder.encode(input_image, encoder_noise)

            # Add noise to the latents
            self.sampler.set_noise_strength(int(image_strength))

            latents = self.sampler.add_noise(latents, self.sampler.timesteps[0])

        else:
            latents = self.sampler.sample_prior(
                (n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype
            )

            # Visualize the latent space
            visualize_tensor(latents, normalize=True)

        if input_image is not None:
            # Unet expects: B, H, W, C = x.shape
            latents = mx.transpose(latents, (0, 3, 1, 2))

        # Perform the denoising loop
        st.text("Denoising Latent Space")
        latent_image_placeholder = st.empty()

        timesteps = tqdm(self.sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            # (1, 320)
            # time_embedding = get_time_embedding(timestep)

            model_input = mx.concatenate([latents] * 2, axis=0) if cfg_weight > 1 else latents

            # Generate noise prediction
            # unet(x, timestep, encoder_x, attn_mask=None, encoder_attn_mask=None)

            model_ouput = self.unet(model_input, timestep=timestep, encoder_x=conditioning)

            if cfg_weight > 1:
                eps_text, eps_neg = model_ouput.split(2)
                model_ouput = eps_neg + cfg_weight * (eps_text - eps_neg)

            # Perform the denoising step
            latents = self.sampler.step(timestep, model_ouput)
            model_input = latents

            # Visualize the latent space
            visualize_tensor(latents, normalize=True, placeholder=latent_image_placeholder)
            yield latents

    def decode(self, latents):
        x = self.autoencoder.decode(latents / self.autoencoder.scaling_factor)
        x = mx.minimum(1, mx.maximum(0, x / 2 + 0.5))

        return x

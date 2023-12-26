# Copyright © 2023 Apple Inc.

import time
from typing import Tuple
from PIL import Image
import numpy as np
import mlx.core as mx
import streamlit as st

from .model_io import (
    load_unet,
    load_text_encoder,
    load_autoencoder,
    load_diffusion_config,
    load_tokenizer,
    _DEFAULT_MODEL,
)
from .sampler import SimpleEulerSampler


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
        self.sampler = SimpleEulerSampler(self.diffusion_config)
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
            seed=None,
            denoising_strength=0.7,
    ):

        # Set the PRNG state
        seed = seed or int(time.time())
        mx.random.seed(seed)

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
            # Convert the input image to a mlx array
            input_image = input_image.resize((512, 512))
            input_image = mx.array(np.array(input_image))

            # Visualize the input image
            input_image_np = np.array(input_image.tolist())
            st.image(input_image_np.squeeze(), caption='Resized Input Image')

            input_image = mx.clip(input_image / 127.5 - 1, -1, 1)
            input_image = mx.expand_dims(input_image, axis=0)

            # Visualize the input image
            input_image_np = np.array(input_image.tolist())
            # Normalize the input image to the range [0, 1]
            input_image_np = (input_image_np + 1) / 2
            st.image(input_image_np.squeeze(), caption='Normalized Input Image')

            # Add noise to the input image
            input_image = self.autoencoder.encode(input_image)
            noise = mx.random.normal(shape=input_image.shape, dtype=input_image.dtype)
            input_image += noise

            # Normalize the input image to the range [0, 1]
            input_image_np = np.array(input_image.tolist())
            input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())

            # Visualize the input image
            st.image(input_image_np.squeeze(), caption='Noised Input Image After 1st Transpose')

            # Rearrange the dimensions to [B, C, H, W]
            input_image = mx.transpose(input_image, (0, 3, 1, 2))

            # # Rearrange the dimensions to [B, H, W, C] - Autoencoder expects: B, H, W, C = x.shape
            input_image = mx.transpose(input_image, (0, 2, 3, 1))

            # Visualize the input image
            input_image_np = np.array(input_image.tolist())
            input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())

            # Visualize the input image
            st.image(input_image_np.squeeze(), caption='Noised Input Image After 2nd Transpose')

            # Use the noisy image as the latent variable

            latents = []
            for _ in range(n_images):
                # latent = self.sampler.add_noise(input_image, mx.ones(n_images))
                latent = input_image
                latents.append(latent)
            latents = mx.concatenate(latents, axis=0)

            # Visualize the latent space
            latents_np = np.array(latents.tolist())
            latents_np = (latents_np - latents_np.min()) / (latents_np.max() - latents_np.min())
            st.text("Latent Space")
            st.image(latents_np.squeeze())

        else:
            latents = self.sampler.sample_prior(
                (n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype
            )

            # Visualize the latent space
            latents_np = np.array(latents.tolist())
            latents_np = (latents_np - latents_np.min()) / (latents_np.max() - latents_np.min())
            st.text("Latent Space")
            st.image(latents_np.squeeze())

        # Perform the denoising loop
        x_t = latents
        for t, t_prev in self.sampler.timesteps(num_steps, dtype=self.dtype):
            x_t_unet = mx.concatenate([x_t] * 2, axis=0) if cfg_weight > 1 else x_t
            t_unet = mx.broadcast_to(t, [len(x_t_unet)])
            eps_pred = self.unet(x_t_unet, t_unet, encoder_x=conditioning)

            if cfg_weight > 1:
                eps_text, eps_neg = eps_pred.split(2)
                eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)

            x_t_prev = self.sampler.step(eps_pred, x_t, t, t_prev)
            x_t = x_t_prev
            yield x_t

    def decode(self, x_t):
        x = self.autoencoder.decode(x_t / self.autoencoder.scaling_factor)
        x = mx.minimum(1, mx.maximum(0, x / 2 + 0.5))
        return x

# Copyright © 2023 Apple Inc.

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
        self.sampler = SimpleEulerSampler(self.diffusion_config)
        self.tokenizer = load_tokenizer(model)

    def _get_text_conditioning(
        self,
        text: str,
        n_images: int = 1,
        cfg_weight: float = 7.5,
        negative_text: str = "",
    ):
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

        return conditioning

    def _denoising_step(self, x_t, t, t_prev, conditioning, cfg_weight: float = 7.5):
        x_t_unet = mx.concatenate([x_t] * 2, axis=0) if cfg_weight > 1 else x_t
        t_unet = mx.broadcast_to(t, [len(x_t_unet)])
        eps_pred = self.unet(x_t_unet, t_unet, encoder_x=conditioning)

        if cfg_weight > 1:
            eps_text, eps_neg = eps_pred.split(2)
            eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)

        x_t_prev = self.sampler.step(eps_pred, x_t, t, t_prev)

        return x_t_prev

    def _denoising_loop(self, x_T, T, conditioning, num_steps: int = 50, cfg_weight: float = 7.5, latent_image_placeholder=None):
        x_t = x_T
        for t, t_prev in self.sampler.timesteps(num_steps, start_time=T, dtype=self.dtype):
            x_t = self._denoising_step(x_t, t, t_prev, conditioning, cfg_weight)

            # Visualize the latent space
            visualize_tensor(x_t, normalize=True, placeholder=latent_image_placeholder)

            yield x_t

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
            denoising_strength=0.7,
    ):

        # Set the PRNG state

        mx.random.seed(int(seed))

        debug_print(f"Seed: {seed}")

        # Get the text conditioning
        conditioning = self._get_text_conditioning(text, n_images, cfg_weight, negative_text)

        if input_image is not None:
            # Define the num steps and start step
            start_step = self.sampler.max_time * denoising_strength
            num_steps = int(num_steps * denoising_strength)

            # Resize the input image
            input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            input_image = mx.array(np.array(input_image))
            input_image = normalize_tensor(input_image.astype(mx.float32), (0, 255), (-1, 1))

            # Get the latents from the input image and add noise according to the
            # start time.
            x_0, _ = self.autoencoder.encode(input_image[None])
            x_0 = mx.broadcast_to(x_0, [n_images] + x_0.shape[1:])
            x_t = self.sampler.add_noise(x_0, mx.array(start_step))

        else:
            # sample_prior(self, shape, dtype=mx.float32, key=None):
            x_t = self.sampler.sample_prior(
                (n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype
            )

        # Visualize the latent space
        st.text("Starting Latent Space")
        visualize_tensor(x_t, normalize=True)

        # Perform the denoising loop

        st.text("Denoising Latent Space")
        latent_image_placeholder = st.empty()

        if input_image is not None:
            yield from self._denoising_loop(x_t, start_step, conditioning, num_steps, cfg_weight, latent_image_placeholder)
        else:
            yield from self._denoising_loop(x_t, self.sampler.max_time, conditioning, num_steps, cfg_weight, latent_image_placeholder)

    def decode(self, x_t):
        x = self.autoencoder.decode(x_t)
        x = mx.minimum(1, mx.maximum(0, x / 2 + 0.5))
        return x

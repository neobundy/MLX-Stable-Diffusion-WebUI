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

from utils import debug_print, normalize_tensor, tensor_head, visualize_tensor

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
            # Convert the input image to a mlx array
            input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            input_image = mx.array(np.array(input_image))
            input_image = normalize_tensor(input_image, (input_image.min(), input_image.max()), (-1, 1))
            input_image = mx.expand_dims(input_image, axis=0)

            debug_print(f"Input image before encoding (first 50 values): {tensor_head(input_image, 50)}")
            # Encode the input image to get the latent representation
            latents = self.autoencoder.encode(input_image)
            latents = normalize_tensor(latents, (latents.min(), latents.max()), (-1, 1))

            debug_print(f"Latent after encoding (first 50 values): {tensor_head(latents, 50)}")

            # Initialize an empty list to store the latents for each image
            latents_list = []
            captions = []
            image_number = 1
            # Normalize the image_strength to the range [-1, 1]
            image_strength = 2 * (image_strength - 0.5)
            for _ in range(n_images):
                # Generate a tensor of random values with the same shape as the latent representation
                encoder_noise = mx.random.normal(shape=latents.shape, dtype=self.dtype)
                # encoder_noise = mx.random.uniform(low=-1.0, high=1.0, shape=latents.shape, dtype=self.dtype)
                encoder_noise = normalize_tensor(encoder_noise, (encoder_noise.min(), encoder_noise.max()), (-1, 1))
                encoder_noise *= image_strength

                debug_print(f"Encoder noise (first 50 values): {tensor_head(encoder_noise, 50)}")

                # Add the noise to the latent representation
                noisy_latents = latents + encoder_noise
                latents_list.append(noisy_latents)
                captions.append(f"Input Image {image_number} with Noise")
                image_number += 1

                debug_print(f"Latents after adding noise (first 50 values): {tensor_head(noisy_latents, 50)}")

            # Create columns for the grid
            cols = st.columns(n_images)
            # Visualize the input image with noise
            for i in range(n_images):
                with cols[i]:
                    visualize_tensor(latents_list[i], caption=captions[i], normalize=True)

            # Duplicate the noisy latents
            latents = mx.concatenate(latents_list, axis=0)

            # Rearrange the dimensions to [B, H, W, C] - Autoencoder expects: B, H, W, C = x.shape
            latents = mx.transpose(latents, (0, 2, 3, 1))

            # Visualize the latent space
            st.text("Latent Space")
            latent_image = mx.transpose(latents, (0, 3, 1, 2))
            visualize_tensor(latent_image, normalize=True)

        else:
            #     def sample_prior(self, shape, dtype=mx.float32, key=None):
            latents = self.sampler.sample_prior(
                (n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype
            )

            # Visualize the latent space
            visualize_tensor(latents, normalize=True)

        x_t = latents
        if input_image is not None:
            # Unet expects: B, H, W, C = x.shape
            x_t = mx.transpose(x_t, (0, 3, 1, 2))

        debug_print(f"the final shape of latents: {x_t.shape}")
        # Perform the denoising loop
        latent_image_placeholder = st.empty()

        # x_t: latent tensor at timestep t
        # t: current timestep mx.array (1000, 980, 960... when num_steps is 50)
        # t_prev: previous timestep mx.array (980, 960, 940... when num_steps is 50)
        # sampler.timesteps(num_steps, dtype=self.dtype) returns a generator
        for t, t_prev in self.sampler.timesteps(num_steps, dtype=self.dtype):

            # Expand the latent tensor and timestep for UNet input
            x_t_unet = mx.concatenate([x_t] * 2, axis=0) if cfg_weight > 1 else x_t
            t_unet = mx.broadcast_to(t, [len(x_t_unet)])

            # Generate noise prediction
            # unet(x, timestep, encoder_x, attn_mask=None, encoder_attn_mask=None)
            # x_t_unet: latent tensor at timestep t
            # t_unet: current timestep mx.array (1000, 980, 960... when num_steps is 50)
            # encoder_x: text encoder output
            eps_pred = self.unet(x_t_unet, t_unet, encoder_x=conditioning)
            # Adjust noise prediction based on cfg_weight
            if cfg_weight > 1:
                eps_text, eps_neg = eps_pred.split(2)
                eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)

            # Perform the denoising step
            x_t_prev = self.sampler.step(eps_pred, x_t, t, t_prev)
            x_t = x_t_prev

            # Visualize the latent space
            visualize_tensor(x_t, normalize=True, placeholder=latent_image_placeholder)
            yield x_t

    def decode(self, x_t):
        x = self.autoencoder.decode(x_t / self.autoencoder.scaling_factor)
        x = mx.minimum(1, mx.maximum(0, x / 2 + 0.5))

        return x

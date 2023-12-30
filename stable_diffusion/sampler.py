# Copyright © 2023 Apple Inc.

from .config import DiffusionConfig

import mlx.core as mx


def _linspace(a, b, num):
    x = mx.arange(0, num) / (num - 1)
    return (b - a) * x + a


def _interp(y, x_new):
    """Interpolate the function defined by (arange(0, len(y)), y) at positions x_new."""
    # x_new is an mx array containing all the timesteps: 1000, 980, 960, 940... if num_steps is 50.
    x_low = x_new.astype(mx.int32)
    x_high = mx.minimum(x_low + 1, len(y) - 1)

    y_low = y[x_low]
    y_high = y[x_high]
    delta_x = x_new - x_low
    y_new = y_low * (1 - delta_x) + delta_x * y_high

    return y_new


class SimpleEulerSampler:
    """A simple Euler integrator that can be used to sample from our diffusion models.

    The method ``step()`` performs one Euler step from x_t to x_t_prev.
    """

    def __init__(self, config: DiffusionConfig):
        # Compute the noise schedule
        if config.beta_schedule == "linear":
            betas = _linspace(
                config.beta_start, config.beta_end, config.num_train_steps
            )
        elif config.beta_schedule == "scaled_linear":
            betas = _linspace(
                config.beta_start**0.5, config.beta_end**0.5, config.num_train_steps
            ).square()
        else:
            raise NotImplementedError(f"{config.beta_schedule} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = mx.cumprod(alphas)

        self._sigmas = mx.concatenate(
            [mx.zeros(1), ((1 - alphas_cumprod) / alphas_cumprod).sqrt()]
        )

    def sample_prior(self, shape, dtype=mx.float32, key=None):
        noise = mx.random.normal(shape, key=key)
        return (
            noise * self._sigmas[-1] * (self._sigmas[-1].square() + 1).rsqrt()
        ).astype(dtype)

    def sigmas(self, t):
        return _interp(self._sigmas, t)

    def timesteps(self, num_steps: int, dtype=mx.float32):
        steps = _linspace(len(self._sigmas) - 1, 0, num_steps + 1).astype(dtype)
        return list(zip(steps, steps[1:]))

    def step(self, eps_pred, x_t, t, t_prev):
        sigma = self.sigmas(t).astype(eps_pred.dtype)
        sigma_prev = self.sigmas(t_prev).astype(eps_pred.dtype)

        dt = sigma_prev - sigma
        x_t_prev = (sigma.square() + 1).sqrt() * x_t + eps_pred * dt

        x_t_prev = x_t_prev * (sigma_prev.square() + 1).rsqrt()

        return x_t_prev


    # TODO: Add noise strength to the sampler
    # def set_noise_strength(self, num_steps: int, noise_strength=1):
    #     """
    #     Set how much noise to add to the input image.
    #     More noise (strength ~ 1) means that the output will be further from the input image.
    #     Less noise (strength ~ 0) means that the output will be closer to the input image.
    #     """
    #     # start_step is the number of noise levels to skip
    #     start_step = num_steps - int(num_steps * noise_strength)
    #     self.timesteps = self.timesteps[start_step:]
    #     self.start_step = start_step

    # TODO: Add noise
    # def add_noise(self, original_samples, timesteps):
    #     alphas_cumprod = self._sigmas.square()
    #     timesteps = mx.array(timesteps)
    #
    #     sqrt_alpha_prod = mx.sqrt(alphas_cumprod[timesteps])
    #     sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    #     while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
    #         sqrt_alpha_prod = sqrt_alpha_prod.expand_dims(-1)
    #
    #     sqrt_one_minus_alpha_prod = mx.sqrt(1 - alphas_cumprod[timesteps])
    #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    #     while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
    #         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.expand_dims(-1)
    #
    #     noise = mx.random.normal(0, 1, original_samples.shape)
    #     noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    #     return noisy_samples

# Copyright © 2023 Apple Inc.

from .config import DiffusionConfig
import numpy as np
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
        self.alphas = alphas
        alphas_cumprod = mx.cumprod(alphas)
        self.alphas_cumprod = alphas_cumprod

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

    def get_timesteps(self, num_steps: int, dtype=mx.float32):
        steps = _linspace(len(self._sigmas) - 1, 0, num_steps + 1).astype(dtype)
        return list(zip(steps, steps[1:]))

    def step(self, eps_pred, x_t, t, t_prev):
        sigma = self.sigmas(t).astype(eps_pred.dtype)
        sigma_prev = self.sigmas(t_prev).astype(eps_pred.dtype)

        dt = sigma_prev - sigma
        x_t_prev = (sigma.square() + 1).sqrt() * x_t + eps_pred * dt

        x_t_prev = x_t_prev * (sigma_prev.square() + 1).rsqrt()

        return x_t_prev


class DDPMSampler(SimpleEulerSampler):
    def __init__(self, config: DiffusionConfig):
        """
        The DDPMSampler class, a subclass of SimpleEulerSampler, is designed to sample from Denoising Diffusion Probabilistic Models (DDPM) during the denoising loop.
        This class is an adaptation of the implementation found in the PyTorch-based repository by Umar Jamil (hkproj):

        https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/ddpm.py

        Parameters:
        config (DiffusionConfig): An instance of the DiffusionConfig class that holds the configuration settings for the diffusion process, such as the beta schedule, beta start and end values, and the number of training steps.
        """

        # Call the constructor of the parent class
        super().__init__(config)

        # Store the total number of training steps
        self.num_train_timesteps = config.num_train_steps

        # Generate an array of timesteps in reverse order
        self.timesteps = mx.arange(0, self.num_train_timesteps)[::-1]

        # Set the number of inference steps
        self.num_inference_steps = 50

        # Initialize the starting step
        self.start_step = 0

        # Create an mlx array with a single element 1.0
        self.one = mx.array(1.0)

    def set_inference_steps(self, num_inference_steps: int = 50):
        """
        This function sets the number of inference steps for the diffusion process. The inference steps are the steps
        at which the noise will be added to the samples during the denoising loop.

        Parameters:
        num_inference_steps (int, optional): The number of inference steps. Default is 50.

        """
        # Set the number of inference steps
        self.num_inference_steps = num_inference_steps

        # Calculate the ratio of the total number of training steps to the number of inference steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps

        # Generate an array of timesteps based on the number of inference steps and the step ratio
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        # Convert the timesteps to an mlx array and store it
        self.timesteps = mx.array(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        This function calculates the previous timestep in the diffusion process.

        Parameters:
        timestep (int): The current timestep in the diffusion process.

        Returns:
        prev_t (int): The previous timestep in the diffusion process.
        """
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t

    def _get_variance(self, timestep: int) -> mx.array:
        """
        This function calculates the variance for a specific timestep in the diffusion process.
        The variance is used to scale the noise that is added to the samples.

        Parameters:
        timestep (int): The current timestep in the diffusion process.

        Returns:
        variance (mx.array): The calculated variance for the current timestep.
        """

        # Get the previous timestep
        prev_t = self._get_previous_timestep(timestep)

        # Get the cumulative product of the alpha values up to the current and previous timestep
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # Calculate the beta value for the current timestep
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # resources/Denoising-Diffusion-Probablilistic-Models.pdf
        # Calculate the variance for the current timestep.
        # alpha_prod_t_prev and alpha_prod_t are the cumulative product of the alpha values up to the previous and current timestep, respectively.
        # current_beta_t is the beta value for the current timestep.
        # The formula is derived from the equations in the DDPM paper: 6, 7.

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Ensure that the variance is not too small.
        # The mx.core.clip() function is used to limit the values of the variance tensor.
        # Any value less than 1e-20 is set to 1e-20. This is done to prevent numerical instability issues that can occur when the variance is too close to zero.
        variance = variance.clip(min=1e-20)

        return variance

    def set_noise_strength(self, num_steps: int, noise_strength=1):
        """
        This function sets the noise strength for the diffusion process. The noise strength is a value between 0 and 1
        that determines the proportion of the total number of steps that will be used to add noise to the original samples.
        A higher noise strength means more steps will be used, resulting in noisier samples.

        Parameters:
        num_steps (int): The total number of steps in the diffusion process.
        noise_strength (float, optional): The strength of the noise to be added. Default is 1, which means all steps will be used.
        """
        # Calculate the starting step based on the noise strength
        start_step = num_steps - int(num_steps * noise_strength)

        # Update the timesteps to start from the calculated step
        self.timesteps = self.timesteps[start_step:]

        # Store the starting step
        self.start_step = start_step

    def add_noise(self, x, timesteps):
        """
        This function adds noise to the input tensor 'x' based on the provided timesteps.
        The noise is generated based on the square of the sigmas and the timesteps.

        Parameters:
        x (mx.ndarray): The input tensor to which noise will be added.
        timesteps (list): The timesteps at which the noise will be added.

        Returns:
        noisy_samples (mx.ndarray): The input tensor 'x' with added noise.
        """

        # Square the sigmas
        alphas_cumprod = self._sigmas.square()

        # Convert the timesteps to an mlx array
        timesteps = mx.array(timesteps)

        # Compute the square root of the product of alphas for the given timesteps
        sqrt_alpha_prod = mx.sqrt(alphas_cumprod[timesteps])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # Expand the dimensions of sqrt_alpha_prod to match the dimensions of 'x'
        while len(sqrt_alpha_prod.shape) < len(x.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.expand_dims(-1)

        # Compute the square root of 1 minus the product of alphas for the given timesteps
        sqrt_one_minus_alpha_prod = mx.sqrt(1 - alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # Expand the dimensions of sqrt_one_minus_alpha_prod to match the dimensions of 'x'
        while len(sqrt_one_minus_alpha_prod.shape) < len(x.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.expand_dims(-1)

        # Generate noise with the same shape as 'x'
        noise = mx.random.normal(0, 1, x.shape)

        # Add the noise to 'x'
        noisy_samples = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def step(self, timestep: int, latents: mx.array, model_output: mx.array):
        """
        This function performs one step of the denoising loop in the Denoising Diffusion Probabilistic Models (DDPM).
        It calculates the predicted previous sample by adding noise to the current sample and the predicted original sample.
        The noise is scaled by the square root of the variance, which is calculated based on the alpha and beta values for the current and previous timesteps.

        Parameters:
        timestep (int): The current timestep in the diffusion process.
        latents (mx.array): The current sample in the diffusion process.
        model_output (mx.array): The predicted noise for the current sample.

        Returns:
        pred_prev_sample (mx.array): The predicted previous sample in the diffusion process.
        """

        # Get the current and previous timestep
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # Compute the cumulative product of the alpha values up to the current and previous timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # Compute the beta values for the current and previous timestep
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute the alpha and beta values for the current timestep
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute the predicted original sample from the predicted noise
        pred_original_sample = (latents - mx.sqrt(beta_prod_t) * model_output) / mx.sqrt(alpha_prod_t)

        # Compute the coefficients for the predicted original sample and the current sample
        pred_original_sample_coeff = (mx.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = mx.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t

        # Compute the predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # Add noise to the predicted previous sample
        variance = 0
        if t > 0:
            # Generate noise with the same shape as the model output
            noise = mx.random.normal(0, 1, model_output.shape)

            # Compute the variance for the current timestep and scale the noise by the square root of the variance
            variance = (self._get_variance(t) ** 0.5) * noise

        # Add the scaled noise to the predicted previous sample
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

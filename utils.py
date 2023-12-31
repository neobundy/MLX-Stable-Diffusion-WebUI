import numpy as np
import mlx.core as mx
import streamlit as st
from safetensors import safe_open as safetensor_open
from pathlib import Path
import shutil

debug = True  # Set this to False if you don't want to print debug messages
LOGFILE = 'log.txt'


def get_state_dict_from_safetensor(checkpoint_path: str):
    """Return the state_dict from the checkpoint."""
    state_dict = {}
    with safetensor_open(checkpoint_path, framework="numpy") as f:
        # Access the data in the file
        for key in f.keys():
            tensor = f.get_tensor(key)
            state_dict[key] = tensor
    return state_dict


def _state_dict(model):
    """Return the model's state_dict as a dictionary."""
    state_dict = {}
    for name, param in model.parameters().items():
        state_dict[name] = param
    return state_dict


def debug_print(*args, **kwargs):
    if debug:
        # Convert the arguments to a string
        message = ' '.join(map(str, args))

        # Print the message to the console
        print(message, **kwargs)


def visualize_tensor(x: mx.array, caption: str = "", normalize:bool = False, placeholder: st.empty = None):
    # Convert the mlx array to a numpy array
    x_np = np.array(x)

    epsilon = 1e-7  # small constant

    # Normalize or scale the tensor
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + epsilon) if normalize else (x_np + 1) / 2

    # Squeeze the tensor to remove single-dimensional entries from the shape
    x_np = x_np.squeeze()

    # Display the image with or without a placeholder
    display_function = placeholder.image if placeholder is not None else st.image
    display_function(x_np, caption=caption if caption else None)


def normalize_tensor(x, old_range, new_range, clip=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clip:
        x = x.clip(new_min, new_max)
    return x


def tensor_head(x, n):
    return x[:n]


def inspect_tensor(x, num_values, header=""):
    if header:
        header = f"{header} - "
    debug_print(f"{header}The first {num_values} value(s) of the tensor with shape {x.shape}: {tensor_head(x, num_values)}")


def get_time_embedding(timestep):
    # Calculate the frequencies. The shape of the resulting tensor is (160,).
    # We use the formula 1 / (10000 ** (i / 160)) for i in range(160).
    freqs = 1 / (10000 ** (mx.arange(start=0, stop=160, dtype=mx.float32) / 160))

    # Create a tensor with the timestep value and expand its dimensions to match the shape of freqs.
    # The shape of the resulting tensor is (1, 160).
    x = mx.array([timestep], dtype=mx.float32).expand_dims(axis=-1) * freqs.expand_dims(axis=0)

    # Concatenate the cosine and sine of x along the last dimension.
    # The shape of the resulting tensor is (1, 160 * 2).
    return mx.concatenate(mx.cos(x), mx.sin(x), dim=-1)


import subprocess


def run_conversion_script(old_checkpoint_path: str, new_checkpoint_path: str):

    """Delete the directory if it exists."""
    new_checkpoint_path = Path(new_checkpoint_path)
    if new_checkpoint_path.exists() and new_checkpoint_path.is_dir():
        shutil.rmtree(new_checkpoint_path)

    command = [
        "python",
        "./convert_original_stable_diffusion_to_diffusers.py",
        "--checkpoint_path", f"{old_checkpoint_path}",
        "--from_safetensors",
        "--to_safetensors",
        "--dump_path", f"{new_checkpoint_path}"
    ]
    subprocess.run(command)


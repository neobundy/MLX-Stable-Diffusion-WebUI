import numpy as np
import mlx.core as mx
import streamlit as st


debug = True  # Set this to False if you don't want to print debug messages
logfile = 'log.txt'

def debug_print(*args, **kwargs):
    if debug:
        # Convert the arguments to a string
        message = ' '.join(map(str, args))

        # Print the message to the console
        print(message, **kwargs)

        # Open the log file in append mode and write the message
        with open(logfile, 'a') as f:
            f.write(message + '\n')

def visualize_tensor(x: mx.array, caption: str = "", normalize:bool = False, placeholder: st.empty = None):
    # Convert the mlx array to a numpy array
    x_np = np.array(x)

    # Normalize or scale the tensor
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min()) if normalize else (x_np + 1) / 2

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

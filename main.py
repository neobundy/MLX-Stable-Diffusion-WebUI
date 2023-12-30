import streamlit as st
from PIL import Image
from tqdm import tqdm
import numpy as np
import time

import mlx.core as mx

from stable_diffusion import StableDiffusion

from stable_diffusion.models import _AVAILABLE_MODELS

from pathlib import Path

from utils import debug_print, normalize_tensor, tensor_head, visualize_tensor


_OUTPUT_FOLDER = Path("output")
_DEFAULT_OUTPUT = "output.png"
_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

_THUMB = "thumb.jpeg"
ORIGINAL_REPO = "https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion"
REPO_URL = "https://github.com/neobundy/MLX-Stable-Diffusion-WebUI"
WHOAMI = "https://x.com/WankyuChoi"

selected_model = st.sidebar.selectbox('Select a model', _AVAILABLE_MODELS, index=0)

st.sidebar.title("Options")

# Add the image uploader widget to the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image for I2I", type=["jpg", "jpeg", "png"])
denoising_strength = st.sidebar.slider("Denoising Strength", min_value=0.1, max_value=1.0, value=0.7)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(input_image, caption='Uploaded Image', use_column_width=True)
else:
    input_image = None

# Add a text box and a button to recall the seed number
seed_number = "-1" if 'seed' not in st.session_state else str(st.session_state['seed'])
seed_number = int(st.sidebar.text_input("Seed Number", value=seed_number))

if st.sidebar.button("Random Seed Number"):
    seed_number = "-1"
    st.session_state['seed'] = seed_number
    st.experimental_rerun()
if st.sidebar.button("Recall Seed Number"):
    if 'seed' in st.session_state:
        seed_number = str(st.session_state['seed'])
    else:
        seed_number = "-1"
    st.session_state['seed'] = seed_number

prompt = st.sidebar.text_input("Prompt")
negative_prompt = st.sidebar.text_input("Negative Prompt")

st.title("Apple MLX Stable Diffusion WebUI")

n_images = st.sidebar.slider("Number of images", min_value=1, max_value=10, value=4)
n_rows = st.sidebar.slider("Number of rows", min_value=1, max_value=10, value=2)
steps = st.sidebar.slider("Number of steps", min_value=1, max_value=100, value=50)
cfg = st.sidebar.slider("CFG", min_value=1.0, max_value=20.0, value=7.5)
decoding_batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=10, value=1)
output = st.sidebar.text_input("Output file name", value=_DEFAULT_OUTPUT)

st.sidebar.caption(ORIGINAL_REPO)
st.sidebar.caption(REPO_URL)
st.sidebar.caption(WHOAMI)

if st.button("Generate"):
    # Retrieve the seed number from the session state
    seed_number = st.session_state.get('seed', None)

    # If the seed number is not set in the session state, generate a new one
    if seed_number is None or int(seed_number) == -1:
        seed_number = int(time.time())
        st.session_state['seed'] = seed_number

    debug_print("====" * 10)
    # print current date and time to the console
    session_type = "Text to Image" if input_image is None else "Image to Image"
    debug_print(f"A New {session_type} Session Started: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    debug_print("====" * 10)
    st.text(f"{session_type} Session with seed: {seed_number}")
    st.text("Model: " + selected_model)
    sd = StableDiffusion(selected_model)

    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        prompt,
        input_image=input_image,
        n_images=n_images,
        cfg_weight=cfg,
        num_steps=steps,
        seed=int(seed_number),
        negative_text=negative_prompt,
        denoising_strength=denoising_strength,
    )

    progress_bar = st.progress(0)
    if input_image is not None:
        total_steps = int(steps * denoising_strength)
        debug_print(f"Total steps: {total_steps}")
    else:
        total_steps = steps

    x_t = None
    for i, x_t in enumerate(tqdm(latents, total=total_steps)):
        mx.simplify(x_t)
        mx.simplify(x_t)
        mx.eval(x_t)
        progress_bar.progress((i + 1) / total_steps)

    # Decode them into images
    if x_t is None:
        st.error("No images generated! Increase the number of steps or decrease the denoising strength.")
    else:
        decoded = []
        for i in tqdm(range(0, n_images, decoding_batch_size)):
            decoded_latents = sd.decode(x_t[i : i + decoding_batch_size])
            decoded.append(decoded_latents)
            mx.eval(decoded[-1])

        # If n_images is not a multiple of n_rows, pad the decoded list with empty images
        if n_images % n_rows != 0:
            for _ in range(n_rows - (n_images % n_rows)):
                decoded.append(mx.zeros_like(decoded[0]))

        # Arrange them on a grid
        x = mx.concatenate(decoded, axis=0)
        x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
        B, H, W, C = x.shape
        x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(n_rows * H, B // n_rows * W, C)
        x = (x * 255).astype(mx.uint8)

        # Save them to disc
        im = Image.fromarray(x.__array__())
        im.save(_OUTPUT_FOLDER / output)

        # Display each image separately
        # Create a list of columns
        columns = st.columns(len(decoded))

        # Display each image separately in a column
        for i, (x, col) in enumerate(zip(decoded, columns)):
            # Squeeze the array to remove extra dimensions
            x = mx.squeeze(x)
            # Scale the pixel values and convert the data type
            x = (x * 255).astype(mx.uint8)

            # Convert the tensor to a numpy array and then to a PIL Image object
            im = Image.fromarray(x.__array__())

            # Display the image in the column using Streamlit
            col.image(im, caption=f'Generated Image {i + 1}')

    if output:
        try:
            image = Image.open(_OUTPUT_FOLDER / output)
            st.image(image, caption='Generated Image')
        except FileNotFoundError:
            st.image(_THUMB, caption='Placeholder Image')
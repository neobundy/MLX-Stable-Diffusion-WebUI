import streamlit as st
from PIL import Image
from tqdm import tqdm

import mlx.core as mx

from stable_diffusion import StableDiffusion

from stable_diffusion.models import _AVAILABLE_MODELS

from pathlib import Path

_OUTPUT_FOLDER = Path("output")
_DEFAULT_OUTPUT = "output.png"
_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

_THUMB = "thumb.jpeg"

selected_model = st.sidebar.selectbox('Select a model', _AVAILABLE_MODELS, index=0)

st.sidebar.title("Options")
prompt = st.sidebar.text_input("Prompt")
negative_prompt = st.sidebar.text_input("Negative Prompt")

st.title("Apple MLX Stable Diffusion WebUI")

n_images = st.sidebar.slider("Number of images", min_value=1, max_value=10, value=4)
n_rows = st.sidebar.slider("Number of rows", min_value=1, max_value=10, value=2)
steps = st.sidebar.slider("Number of steps", min_value=1, max_value=100, value=50)
cfg = st.sidebar.slider("CFG", min_value=1.0, max_value=20.0, value=7.5)
decoding_batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=10, value=1)
output = st.sidebar.text_input("Output file name", value=_DEFAULT_OUTPUT)

if st.button("Generate"):

    st.text("Generating images using the checkpoint: " + selected_model)
    sd = StableDiffusion(selected_model)

    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        prompt,
        n_images=n_images,
        cfg_weight=cfg,
        num_steps=steps,
        negative_text=negative_prompt,
    )

    progress_bar = st.progress(0)
    for i, x_t in enumerate(tqdm(latents, total=steps)):
        mx.simplify(x_t)
        mx.simplify(x_t)
        mx.eval(x_t)
        progress_bar.progress((i + 1) / steps)

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, n_images, decoding_batch_size)):
        decoded.append(sd.decode(x_t[i : i + decoding_batch_size]))
        mx.eval(decoded[-1])

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

if output:
    try:
        image = Image.open(_OUTPUT_FOLDER / output)
        st.image(image, caption='Generated Image')
    except FileNotFoundError:
        st.image(_THUMB, caption='Placeholder Image')

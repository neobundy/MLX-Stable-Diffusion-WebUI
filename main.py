import streamlit as st
from PIL import Image
from tqdm import tqdm
import numpy as np
import time
import os
import mlx.core as mx

from stable_diffusion import StableDiffusion, StableDiffusionLocal
from stable_diffusion.models import _AVAILABLE_MODELS
from pathlib import Path
from stable_diffusion.config import DiffuserModelPathConfig
from utils import debug_print, normalize_tensor, tensor_head, visualize_tensor, run_conversion_script

diffuser_models = DiffuserModelPathConfig()

_OUTPUT_FOLDER = Path("output")
_DEFAULT_OUTPUT = "output.png"
_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
SAFETENSORS_MODELS_FOLDER = "./models"
DIFFUSER_MODELS_FOLDER = diffuser_models.model_path

# Get a list of all files in the SAFETENSORS_MODELS_FOLDER directory with the 'safetensors' extension
model_files = [f for f in os.listdir(SAFETENSORS_MODELS_FOLDER) if f.endswith('.safetensors') and 'XL' not in f]
model_files.sort()

_THUMB = "thumb.jpeg"

MAIN_TITLE = "Apple MLX Stable Diffusion WebUI"

ORIGINAL_REPO = "https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion"
REPO_URL = "https://github.com/neobundy/MLX-Stable-Diffusion-WebUI"
WHOAMI = "https://x.com/WankyuChoi"

# Sidebar options
# Create a dropdown list with the model files
USE_HUGGINGFACE = st.sidebar.checkbox('USE HUGGINGFACE', value=False)

if USE_HUGGINGFACE:
    selected_model = st.sidebar.selectbox('Select a model', _AVAILABLE_MODELS, index=0)
else:
    selected_model = st.sidebar.selectbox('Select a model', model_files, index=0)
    selected_model_without_extension = os.path.splitext(selected_model)[0]

    selected_diffuser_model = f"{DIFFUSER_MODELS_FOLDER}/{selected_model_without_extension}"
    if not os.path.exists(selected_diffuser_model):
        st.info("Converting model to diffuser format: " + selected_model)
        run_conversion_script(f"{SAFETENSORS_MODELS_FOLDER}/{selected_model}", selected_diffuser_model)

    selected_model = selected_diffuser_model

st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Choose an image for I2I", type=["jpg", "jpeg", "png"])
denoising_strength = st.sidebar.slider("Denoising Strength", min_value=0.1, max_value=1.0, value=0.7)
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(input_image, caption='Uploaded Image', use_column_width=True)
else:
    input_image = None
seed_number = int(st.sidebar.text_input("Seed Number", value="-1" if 'seed' not in st.session_state else str(st.session_state['seed'])))
if st.sidebar.button("Random Seed Number"):
    seed_number = "-1"
    st.session_state['seed'] = seed_number
    st.experimental_rerun()
if st.sidebar.button("Recall Seed Number"):
    seed_number = "-1" if 'seed' not in st.session_state else str(st.session_state['seed'])
    st.session_state['seed'] = seed_number
prompt = st.sidebar.text_input("Prompt")
negative_prompt = st.sidebar.text_input("Negative Prompt")
n_images = st.sidebar.slider("Number of images", min_value=1, max_value=10, value=4)
n_rows = st.sidebar.slider("Number of rows", min_value=1, max_value=10, value=2)
steps = st.sidebar.slider("Number of steps", min_value=1, max_value=100, value=50)
cfg = st.sidebar.slider("CFG", min_value=1.0, max_value=20.0, value=7.5)
decoding_batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=10, value=1)
output = st.sidebar.text_input("Output file name", value=_DEFAULT_OUTPUT)

# Sidebar captions
st.sidebar.caption(ORIGINAL_REPO)
st.sidebar.caption(REPO_URL)
st.sidebar.caption(WHOAMI)

# Main title
st.title(MAIN_TITLE)

# Generate button
if st.button("Generate"):
    seed_number = int(time.time()) if seed_number == -1 else seed_number
    st.session_state['seed'] = seed_number
    session_type = "Text to Image" if input_image is None else "Image to Image"
    st.text(f"{session_type} Session with seed: {seed_number}")
    st.text("Model: " + selected_model)

    if USE_HUGGINGFACE:
        sd = StableDiffusion(selected_model)
    else:
        sd = StableDiffusionLocal(diffuser_model_path=selected_model)

    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        prompt,
        input_image=input_image,
        n_images=n_images,
        cfg_weight=cfg,
        num_steps=steps,
        seed=seed_number,
        negative_text=negative_prompt,
        denoising_strength=denoising_strength,
    )

    progress_bar = st.progress(0)
    total_steps = int(steps * denoising_strength) if input_image is not None else steps
    x_t = None
    for i, x_t in enumerate(tqdm(latents, total=total_steps)):
        mx.simplify(x_t)
        mx.eval(x_t)
        progress_bar.progress((i + 1) / total_steps)

    # Decode them into images
    if x_t is None:
        st.error("No images generated! Increase the number of steps or decrease the denoising strength.")
    else:
        decoded = [sd.decode(x_t[i : i + decoding_batch_size]) for i in tqdm(range(0, n_images, decoding_batch_size))]
        mx.eval(decoded[-1])

        # If n_images is not a multiple of n_rows, pad the decoded list with empty images
        decoded += [mx.zeros_like(decoded[0])] * (n_rows - (n_images % n_rows)) if n_images % n_rows != 0 else []

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
        columns = st.columns(len(decoded))
        for i, (x, col) in enumerate(zip(decoded, columns)):
            x = mx.squeeze(x)
            x = (x * 255).astype(mx.uint8)
            im = Image.fromarray(x.__array__())
            col.image(im, caption=f'Generated Image {i + 1}')

    if output:
        try:
            image = Image.open(_OUTPUT_FOLDER / output)
            st.image(image, caption='Generated Image')
        except FileNotFoundError:
            st.image(_THUMB, caption='Placeholder Image')
# Copyright © 2023 Apple Inc.

from typing import Optional
import json
from functools import partial

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten
from safetensors import safe_open as safetensor_open

from .clip import CLIPTextModel
from .config import AutoencoderConfig, CLIPTextModelConfig, DiffusionConfig, UNetConfig
from .tokenizer import Tokenizer
from .unet import UNetModel
from .vae import Autoencoder

from .models import _DEFAULT_MODEL, _MODELS
from .config import DiffuserModelPathConfig

from tqdm import tqdm

logfile = 'log.txt'
_DEBUG = False


def _debug_print(*args, **kwargs):
    if _DEBUG:
        # Convert the arguments to a string
        message = ' '.join(map(str, args))

        # Print the message to the console
        print(message, **kwargs)

        # Open the log file in append mode and write the message
        with open(logfile, 'a') as f:
            f.write(message + '\n')


def _from_numpy(x):
    return mx.array(np.ascontiguousarray(x))


# The `map_*_weights` functions are used to adjust the weights of a model when loading it from a file.
# The weights of the model in the file might be in a different format than the weights of the model in the current codebase.
# When you load a pre-trained model, the weights are stored in a dictionary where the keys are the names of the parameters in the model.
# If the architecture of your model is different from the architecture of the model that the weights were trained on, you might need to adjust the keys and/or the weights to match your model's architecture.
# This is what the `map_*_weights` functions are doing. They are adjusting the keys and the weights to match the architecture of the models in the current codebase.
def map_unet_weights(key, value):
    # Map up/downsampling
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
        _debug_print(f"Replaced 'downsamplers.0.conv' with 'downsample' in {key}")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")
        _debug_print(f"Replaced 'upsamplers.0.conv' with 'upsample' in {key}")

    # Map the mid block
    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
        _debug_print(f"Replaced 'mid_block.resnets.0' with 'mid_blocks.0' in {key}")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
        _debug_print(f"Replaced 'mid_block.attentions.0' with 'mid_blocks.1' in {key}")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")
        _debug_print(f"Replaced 'mid_block.resnets.1' with 'mid_blocks.2' in {key}")

    # Map attention layers
    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
        _debug_print(f"Replaced 'to_k' with 'key_proj' in {key}")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
        _debug_print(f"Replaced 'to_out.0' with 'out_proj' in {key}")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
        _debug_print(f"Replaced 'to_q' with 'query_proj' in {key}")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")
        _debug_print(f"Replaced 'to_v' with 'value_proj' in {key}")

    # Map transformer ffn
    if "ff.net.2" in key:
        key = key.replace("ff.net.2", "linear3")
        _debug_print(f"Replaced 'ff.net.2' with 'linear3' in {key}")
    if "ff.net.0" in key:
        k1 = key.replace("ff.net.0.proj", "linear1")
        k2 = key.replace("ff.net.0.proj", "linear2")
        v1, v2 = np.split(value, 2)
        _debug_print(f"Replaced 'ff.net.0.proj' with 'linear1' and 'linear2' in {key}")

        return [(k1, _from_numpy(v1)), (k2, _from_numpy(v2))]

    # The weights of this 1x1 convolutional layer would be a 4-dimensional tensor
    # with shape [out_channels, in_channels, 1, 1].
    # The squeeze() function is used to remove the dimensions of size 1 from this tensor,
    # converting it to a 2-dimensional tensor with shape [out_channels, in_channels].
    # This is because the corresponding layer in the current model might be a linear layer
    # rather than a convolutional layer, and the weights for a linear layer are expected to be a 2-dimensional tensor.

    if "conv_shortcut.weight" in key:
        value = value.squeeze()
        _debug_print(f"Squeezed 'conv_shortcut.weight' in {key}")

    # Transform the weights from 1x1 convs to linear
    if len(value.shape) == 4 and ("proj_in" in key or "proj_out" in key):
        value = value.squeeze()
        _debug_print(f"Squeezed 'proj_in' or 'proj_out' in {key}")

    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        _debug_print(f"Transposed dimensions in {key}")

    return [(key, _from_numpy(value))]


def map_clip_text_encoder_weights(key, value):
    # Remove prefixes
    if key.startswith("text_model."):
        key = key[11:]
        _debug_print(f"Removed 'text_model.' prefix from {key}")
    if key.startswith("embeddings."):
        key = key[11:]
        _debug_print(f"Removed 'embeddings.' prefix from {key}")
    if key.startswith("encoder."):
        key = key[8:]
        _debug_print(f"Removed 'encoder.' prefix from {key}")

    # Map attention layers
    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
        _debug_print(f"Replaced 'self_attn.' with 'attention.' in {key}")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
        _debug_print(f"Replaced 'q_proj.' with 'query_proj.' in {key}")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
        _debug_print(f"Replaced 'k_proj.' with 'key_proj.' in {key}")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")
        _debug_print(f"Replaced 'v_proj.' with 'value_proj.' in {key}")

    # Map ffn layers
    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
        _debug_print(f"Replaced 'mlp.fc1' with 'linear1' in {key}")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")
        _debug_print(f"Replaced 'mlp.fc2' with 'linear2' in {key}")

    return [(key, _from_numpy(value))]


def map_vae_weights(key, value):
    # Map up/downsampling
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
        _debug_print(f"Replaced 'downsamplers.0.conv' with 'downsample' in {key}")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")
        _debug_print(f"Replaced 'upsamplers.0.conv' with 'upsample' in {key}")

    # Map attention layers
    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
        _debug_print(f"Replaced 'to_k' with 'key_proj' in {key}")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
        _debug_print(f"Replaced 'to_out.0' with 'out_proj' in {key}")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
        _debug_print(f"Replaced 'to_q' with 'query_proj' in {key}")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")
        _debug_print(f"Replaced 'to_v' with 'value_proj' in {key}")

    # Map the mid block
    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
        _debug_print(f"Replaced 'mid_block.resnets.0' with 'mid_blocks.0' in {key}")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
        _debug_print(f"Replaced 'mid_block.attentions.0' with 'mid_blocks.1' in {key}")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")
        _debug_print(f"Replaced 'mid_block.resnets.1' with 'mid_blocks.2' in {key}")

    # Map the quant/post_quant layers
    if "quant_conv" in key:
        key = key.replace("quant_conv", "quant_proj")
        value = value.squeeze()
        _debug_print(f"Replaced 'quant_conv' with 'quant_proj' and squeezed value in {key}")

    # Map the conv_shortcut to linear
    if "conv_shortcut.weight" in key:
        value = value.squeeze()
        _debug_print(f"Squeezed 'conv_shortcut.weight' in {key}")

    # Rearrange the dimensions to [B, H, W, C] - Autoencoder expects: B, H, W, C = x.shape
    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        _debug_print(f"Transposed dimensions in {key}")

    return [(key, _from_numpy(value))]


def _flatten(params):
    return [(k, v) for p in params for (k, v) in p]


# The weights of the model can be loaded as 16-bit floating point numbers, which is a form of quantization known as half-precision floating point.
# This can reduce the memory requirements of the model by half compared to 32-bit floating point numbers, at the cost of reduced numerical precision.
def _load_safetensor_weights(mapper, model, weight_file, float16: bool = False):
    dtype = np.float16 if float16 else np.float32

    _debug_print(f"Loading weights from {weight_file}")

    with safetensor_open(weight_file, framework="numpy") as f:
        keys = list(f.keys())
        weights = _flatten([mapper(k, f.get_tensor(k).astype(dtype)) for k in tqdm(keys, desc=f"Loading weights from {weight_file}...")])
    model.update(tree_unflatten(weights))


def _check_key(key: str, part: str):
    if key not in _MODELS:
        raise ValueError(
            f"[{part}] '{key}' model not found, choose one of {{{','.join(_MODELS.keys())}}}"
        )


def load_unet(key: str = _DEFAULT_MODEL, float16: bool = False):
    """Load the stable diffusion UNet from Hugging Face Hub."""
    _check_key(key, "load_unet")

    # Download the config and create the model
    unet_config = _MODELS[key]["unet_config"]
    with open(hf_hub_download(key, unet_config)) as f:
        config = json.load(f)

    n_blocks = len(config["block_out_channels"])
    model = UNetModel(
        UNetConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=[config["layers_per_block"]] * n_blocks,
            num_attention_heads=[config["attention_head_dim"]] * n_blocks
            if isinstance(config["attention_head_dim"], int)
            else config["attention_head_dim"],
            cross_attention_dim=[config["cross_attention_dim"]] * n_blocks,
            norm_num_groups=config["norm_num_groups"],
        )
    )

    # Download the weights and map them into the model
    unet_weights = _MODELS[key]["unet"]
    weight_file = hf_hub_download(key, unet_weights)
    _load_safetensor_weights(map_unet_weights, model, weight_file, float16)

    return model


def load_text_encoder(key: str = _DEFAULT_MODEL, float16: bool = False):
    """Load the stable diffusion text encoder from Hugging Face Hub."""
    _check_key(key, "load_text_encoder")

    # Download the config and create the model
    text_encoder_config = _MODELS[key]["text_encoder_config"]
    with open(hf_hub_download(key, text_encoder_config)) as f:
        config = json.load(f)

    model = CLIPTextModel(
        CLIPTextModelConfig(
            num_layers=config["num_hidden_layers"],
            model_dims=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_length=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
        )
    )

    # Download the weights and map them into the model
    text_encoder_weights = _MODELS[key]["text_encoder"]
    weight_file = hf_hub_download(key, text_encoder_weights)
    _load_safetensor_weights(map_clip_text_encoder_weights, model, weight_file, float16)

    return model


def load_autoencoder(key: str = _DEFAULT_MODEL, float16: bool = False):
    """Load the stable diffusion autoencoder from Hugging Face Hub."""
    _check_key(key, "load_autoencoder")

    # Download the config and create the model
    vae_config = _MODELS[key]["vae_config"]
    with open(hf_hub_download(key, vae_config)) as f:
        config = json.load(f)

    model = Autoencoder(
        AutoencoderConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            latent_channels_out=2 * config["latent_channels"],
            latent_channels_in=config["latent_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
        )
    )

    # Download the weights and map them into the model
    vae_weights = _MODELS[key]["vae"]
    weight_file = hf_hub_download(key, vae_weights)
    _load_safetensor_weights(map_vae_weights, model, weight_file, float16)

    return model


def load_diffusion_config(key: str = _DEFAULT_MODEL):
    """Load the stable diffusion config from Hugging Face Hub."""
    _check_key(key, "load_diffusion_config")

    diffusion_config = _MODELS[key]["diffusion_config"]
    with open(hf_hub_download(key, diffusion_config)) as f:
        config = json.load(f)

    return DiffusionConfig(
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        beta_schedule=config["beta_schedule"],
        num_train_steps=config["num_train_timesteps"],
    )


def load_tokenizer(key: str = _DEFAULT_MODEL):
    _check_key(key, "load_tokenizer")

    vocab_file = hf_hub_download(key, _MODELS[key]["tokenizer_vocab"])
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = hf_hub_download(key, _MODELS[key]["tokenizer_merges"])
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return Tokenizer(bpe_ranks, vocab)


def load_unet_local(weights_path: str, config_path: str, float16: bool = False):
    """Load the stable diffusion UNet from local files."""

    if config_path is None:
        # Download the default config
        key = _DEFAULT_MODEL
        _check_key(key, "load_unet")
        unet_config = _MODELS[key]["unet_config"]
        _debug_print("Unet: Using default config - {unet_config}")
        with open(hf_hub_download(key, unet_config)) as f:
            config = json.load(f)
    else:
        _debug_print(f"Unet: Using config - {config_path}")
        with open(config_path) as f:
            config = json.load(f)

    n_blocks = len(config["block_out_channels"])
    model = UNetModel(
        UNetConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=[config["layers_per_block"]] * n_blocks,
            num_attention_heads=[config["attention_head_dim"]] * n_blocks
            if isinstance(config["attention_head_dim"], int)
            else config["attention_head_dim"],
            cross_attention_dim=[config["cross_attention_dim"]] * n_blocks,
            norm_num_groups=config["norm_num_groups"],
        )
    )

    # Load the weights into the model
    _load_safetensor_weights(map_unet_weights, model, weights_path, float16)

    return model


def load_text_encoder_local(weights_path: str, config_path: str, float16: bool = False):
    """Load the stable diffusion text encoder from local files."""

    if config_path is None:
        # Download the default config
        key = _DEFAULT_MODEL
        _check_key(key, "load_text_encoder")
        text_encoder_config = _MODELS[key]["text_encoder_config"]
        _debug_print("Text Encoder: Using default config - {text_encoder_config}")
        with open(hf_hub_download(key, text_encoder_config)) as f:
            config = json.load(f)
    else:
        _debug_print(f"Text Encoder: Using config - {config_path}")
        with open(config_path) as f:
            config = json.load(f)

    model = CLIPTextModel(
        CLIPTextModelConfig(
            num_layers=config["num_hidden_layers"],
            model_dims=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_length=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
        )
    )

    # Load the weights into the model
    _load_safetensor_weights(map_clip_text_encoder_weights, model, weights_path, float16)

    return model


def load_autoencoder_local(weights_path: str, config_path: str, float16: bool = False):
    """Load the stable diffusion autoencoder from local files."""

    if config_path is None:
        # Download the default config
        key = _DEFAULT_MODEL
        _check_key(key, "load_autoencoder")
        vae_config = _MODELS[key]["vae_config"]
        _debug_print("Autoencoder: Using default config - {vae_config}")
        with open(hf_hub_download(key, vae_config)) as f:
            config = json.load(f)
    else:
        _debug_print(f"Autoencoder: Using config - {config_path}")
        with open(config_path) as f:
            config = json.load(f)

    model = Autoencoder(
        AutoencoderConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            latent_channels_out=2 * config["latent_channels"],
            latent_channels_in=config["latent_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
        )
    )

    # Load the weights into the model
    _load_safetensor_weights(map_vae_weights, model, weights_path, float16)

    return model


def load_diffusion_config_local(config_path:str):

    with open(config_path, "r") as f:
        config = json.load(f)

    return DiffusionConfig(
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        beta_schedule=config["beta_schedule"],
        num_train_steps=config["num_train_timesteps"],
    )


def load_tokenizer_local(vocab_path: str, merges_path: str):

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    with open(merges_path, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return Tokenizer(bpe_ranks, vocab)

def load_diffuser_model(diffuser_model_path: str, float16: bool = False):
    # ./unet/models/model_path

    _debug_print(f"Loading diffuser model from {diffuser_model_path}")
    diffuser_model = DiffuserModelPathConfig(model_path=diffuser_model_path)
    unet = load_unet_local(diffuser_model.unet, diffuser_model.unet_config, float16)
    text_encoder = load_text_encoder_local(diffuser_model.text_encoder, diffuser_model.text_encoder_config, float16)
    autoencoder = load_autoencoder_local(diffuser_model.vae, diffuser_model.vae_config, float16)
    diffusion_config = load_diffusion_config_local(diffuser_model.diffusion_config)
    tokenizer = load_tokenizer_local(diffuser_model.tokenizer_vocab, diffuser_model.tokenizer_merges)

    return { "unet": unet,
             "text_encoder": text_encoder,
             "autoencoder": autoencoder,
             "diffusion_config": diffusion_config,
             "tokenizer": tokenizer }
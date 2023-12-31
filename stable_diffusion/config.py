from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path


class DiffuserModelPathConfig:
    def __init__(self, model_path: str = "./diffuser_models"):
        self.model_path = model_path

    @property
    def unet_config(self):
        return self.model_path + "/unet/config.json"

    @property
    def unet(self):
        return self.model_path + "/unet/diffusion_pytorch_model.safetensors"

    @property
    def scheduler(self):
        return self.model_path + "/scheduler/scheduler_config.json"

    @property
    def text_encoder_config(self):
        return self.model_path + "/text_encoder/config.json"

    @property
    def text_encoder(self):
        return self.model_path + "/text_encoder/model.safetensors"

    @property
    def vae_config(self):
        return self.model_path + "/vae/config.json"

    @property
    def vae(self):
        return self.model_path + "/vae/diffusion_pytorch_model.safetensors"

    @property
    def diffusion_config(self):
        return self.model_path + "/scheduler/scheduler_config.json"

    @property
    def tokenizer_vocab(self):
        return self.model_path + "/tokenizer/vocab.json"

    @property
    def tokenizer_merges(self):
        return self.model_path + "/tokenizer/merges.txt"

@dataclass
class BaseConfig:
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class AutoencoderConfig(BaseConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels_out: int = 8
    latent_channels_in: int = 4
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215

@dataclass
class CLIPTextModelConfig(BaseConfig):
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408

@dataclass
class UNetConfig(BaseConfig):
    in_channels: int = 4
    out_channels: int = 4
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    layers_per_block: Tuple[int] = (2, 2, 2, 2)
    mid_block_layers: int = 2
    transformer_layers_per_block: Tuple[int] = (1, 1, 1, 1)
    num_attention_heads: Tuple[int] = (5, 10, 20, 20)
    cross_attention_dim: Tuple[int] = (1024,) * 4
    norm_num_groups: int = 32

@dataclass
class DiffusionConfig(BaseConfig):
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_steps: int = 1000
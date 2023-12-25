_AVAILABLE_MODELS = [
    "stabilityai/stable-diffusion-2-1-base",
    "Lykon/dreamshaper-8",
    "Lykon/absolute-reality-1.81",
    # "stabilityai/sdxl-turbo",
    # "Lykon/dreamshaper-xl-turbo",
    # "stabilityai/stable-diffusion-xl-base-1.0",
]

_DEFAULT_MODEL = _AVAILABLE_MODELS[0]

def generate_model_dict():
    return {
        "unet_config": "unet/config.json",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder/model.safetensors",
        "vae_config": "vae/config.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "diffusion_config": "scheduler/scheduler_config.json",
        "tokenizer_vocab": "tokenizer/vocab.json",
        "tokenizer_merges": "tokenizer/merges.txt",
    }

_MODELS = {model: generate_model_dict() for model in _AVAILABLE_MODELS}
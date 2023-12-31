from stable_diffusion.config import PathConfig
from stable_diffusion.model_io import preload_models_from_safetensor_weights

from utils import _state_dict

INSPECTION_FILE = "model_inspection.txt"
NUM_ITEMS = 100

MODEL_FILE = "./models/v2-1_512-ema-pruned.safetensors"
MODEL_FILE1 = "./unet/diffusion_pytorch_model_test.safetensors"
MODEL_FILE2 = "./unet/xxmix9realistic_v40.safetensors"


from utils import get_state_dict_from_safetensor

# Recreate the inspection file at every execution of the script
with open(INSPECTION_FILE, 'w') as f:
    pass

def write_to_file(*args, **kwargs):
    """Write the text to the inspection file."""
    # Convert the arguments to a string
    message = ' '.join(map(str, args))

    # Print the message to the console
    print(message, **kwargs)

    # Open the log file in append mode and write the message
    with open(INSPECTION_FILE, 'a') as f:
        f.write(message + '\n')


def inspect_model(path_config: PathConfig, keys_only=True):
    """Inspect the contents of the models."""

    # Load the models using the provided config and weights paths
    unet_model = load_unet_local(path_config.unet_config, MODEL_FILE)
    text_encoder_model = load_text_encoder_local(MODEL_FILE)
    autoencoder_model = load_autoencoder_local(MODEL_FILE)
    diffusion_config = load_diffusion_config_local(path_config.diffusion_config)
    tokenizer = load_tokenizer_local(path_config.tokenizer_vocab, path_config.tokenizer_merges)

    # Convert the models' state_dict to a dictionary and iterate over it
    for model_name, model in zip(["unet", "text_encoder", "autoencoder"], [unet_model, text_encoder_model, autoencoder_model]):
        write_to_file("-" * 50)
        write_to_file(f"Model: {model_name}")
        write_to_file("-" * 50)
        for key, value in _state_dict(model).items():
            # Print the key and the corresponding tensor
            if keys_only:
                write_to_file(key)
            else:
                write_to_file(f"{key}: {value}")

    # Print the diffusion config
    write_to_file("-" * 50)
    write_to_file("Diffusion Config:", diffusion_config)

    # Print the tokenizer vocab and merges

    write_to_file("-" * 50)
    write_to_file(f"Tokenizer Ranks{NUM_ITEMS}:", list(tokenizer.bpe_ranks.items())[:NUM_ITEMS])
    write_to_file("-" * 50)
    write_to_file(f"Tokenizer Vocab{NUM_ITEMS}:", list(tokenizer.vocab.items())[:NUM_ITEMS])

    # TODO: inspect the model to see if it has modules baked in


if __name__ == "__main__":
    write_to_file("-" * 50)
    write_to_file("Model Weights File: ", MODEL_FILE1)
    write_to_file("-" * 50)
    # inspect_model(path_config=PathConfig(), keys_only=True)
    state_dict = get_state_dict_from_safetensor(MODEL_FILE1)

    for key, value in state_dict.items():
        write_to_file(f"{key}: {value.shape}")

    write_to_file("after conversion")

    # return { "unet": unet,
    #          "text_encoder": text_encoder,
    #          "autoencoder": autoencoder,
    #          "diffusion_config": diffusion_config,
    #          "tokenizer": tokenizer }
    # models = preload_models_from_safetensor_weights(MODEL_FILE)
    # for model_name, model in models.items():
    #     if model_name != "unet":
    #         print("-" * 50)
    #         print(f"model: {model_name}")
    #         print("-" * 50)
    #         for name, param in model.__dict__.items():
    #             print(f"{name}: {param}")
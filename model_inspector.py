from stable_diffusion.config import PathConfig
from stable_diffusion.model_io import (
    load_unet_local,
    load_text_encoder_local,
    load_autoencoder_local,
    load_diffusion_config_local,
    load_tokenizer_local,
)

from utils import _state_dict

INSPECTION_FILE = "model_inspection.txt"
NUM_ITEMS = 100

MODEL_FILE = "./models/absolutereality_v181.safetensors"

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


if __name__ == "__main__":
    inspect_model(path_config=PathConfig(), keys_only=True)
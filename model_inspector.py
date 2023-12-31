from pathlib import Path
import safetensors

# Specify the path to your safetensor model
model_path = Path('./models')
model_file = 'absolutereality_v181.safetensors'

# Construct the full path to the model file
full_model_path = model_path / model_file

def inspect_model(model: Path, keys_only=True):
    """Inspect the contents of a safetensor model."""

    with safetensors.safe_open(str(model), framework="pytorch") as f:
        # Iterate over the keys in the state dict
        for key in f.keys():
            # Print the key and the corresponding tensor
            if keys_only:
                print(key)
            else:
                print(f"{key}: {f.get_tensor(key)}")


if __name__ == "__main__":
    inspect_model(model=full_model_path, keys_only=True)
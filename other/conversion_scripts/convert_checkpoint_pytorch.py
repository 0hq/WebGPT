import json
import struct
import torch
import os


def save_weights_to_bin_files(checkpoint, folder_name):
    for key, value in checkpoint['model'].items():
        print(f"{key}: {value.shape}")
        if key.startswith('_orig_mod.'):
            continue
        with open(os.path.join(folder_name, f"{key}_gpt.bin"), 'wb') as file:
            for single_value in value.numpy().flatten():
                file.write(struct.pack('<f', single_value))


def save_model_args_to_json(checkpoint, folder_name):
    with open(os.path.join(folder_name, 'params_gpt.json'), 'w') as file:
        json.dump(checkpoint['model_args'], file, indent=4)


def export_pytorch_checkpoint_to_bin_files(ckpt_path, folder_name):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    os.makedirs(folder_name, exist_ok=True)

    save_model_args_to_json(checkpoint, folder_name)
    save_weights_to_bin_files(checkpoint, folder_name)


if __name__ == "__main__":
    ckpt_path = 'other/conversion_scripts/ckpt.pt'
    folder_name = 'model_weights/'

    export_pytorch_checkpoint_to_bin_files(ckpt_path, folder_name)

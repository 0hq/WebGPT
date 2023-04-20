import os
import struct
from transformers import GPT2LMHeadModel


def load_pretrained_gpt_model(model_type):
    print(f"Loading weights from pretrained GPT model: {model_type}")
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    return model_hf


def export_weights_to_files(model, folder_name):
    os.makedirs(folder_name, exist_ok=True)

    state_dict = model.state_dict()

    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")

        with open(os.path.join(folder_name, f"{k}_gpt.bin"), 'wb') as f:
            for value in v.cpu().numpy().flatten():
                f.write(struct.pack('<f', value))


def main(model_type, folder_name):
    model_hf = load_pretrained_gpt_model(model_type)
    export_weights_to_files(model_hf, folder_name)


if __name__ == "__main__":
    model_type = 'gpt2'
    folder_name = f"{model_type}/"

    main(model_type, folder_name)

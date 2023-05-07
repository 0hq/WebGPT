import os
import struct
import json
from transformers import GPT2LMHeadModel


def load_pretrained_gpt_model(model_type):
    print(f"Loading weights from pretrained GPT model: {model_type}")
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    return model_hf


transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
              'mlp.c_fc.weight', 'mlp.c_proj.weight']


def export_weights_to_files(model, folder_name):
    os.makedirs(folder_name, exist_ok=True)

    state_dict = model.state_dict()

    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")

        with open(os.path.join(folder_name, f"{k}_gpt.bin"), 'wb') as f:
            values = v.cpu().numpy()
            # Only use this if using old minGPT model.
            # if any(k.endswith(w) for w in transposed):
            #     values = values.T
            for value in values.flatten():
                f.write(struct.pack('<f', value))


def save_model_args_to_json(model, folder_name):
    with open(os.path.join(folder_name, 'params_gpt.json'), 'w') as f:
        json.dump(model.config.to_dict(), f, indent=4)


def main(model_type, folder_name):
    model_hf = load_pretrained_gpt_model(model_type)
    export_weights_to_files(model_hf, folder_name)
    save_model_args_to_json(model_hf, folder_name)


if __name__ == "__main__":
    model_type = 'gpt2'
    folder_name = f"{model_type}/"

    main(model_type, folder_name)

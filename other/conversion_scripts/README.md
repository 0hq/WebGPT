# Running custom models on WebGPU

It's fairly easy to run custom models on WebGPU. At the moment, I only support PyTorch models via the scripts below but it should be fairly simple to export other model weights to work here.

Importing weights requires you to export transformer weights as a series of individual .bin files. Pardon the somewhat inconvenient process as loading such significant file sizes into Javascript requires some clever engineering.

An example structure with only two layers. Each matrix is collapes into a row-major 1-dimensional array.

```
transformer.wte.weight.bin: [65, 128]
transformer.wpe.weight.bin: [64, 128]
transformer.h.0.ln_1.weight.bin: [128]
transformer.h.0.ln_1.bias.bin: [128]
transformer.h.0.attn.c_attn.weight.bin: [384, 128]
transformer.h.0.attn.c_attn.bias.bin: [384]
transformer.h.0.attn.c_proj.weight.bin: [128, 128]
transformer.h.0.attn.c_proj.bias.bin: [128]
transformer.h.0.ln_2.weight.bin: [128]
transformer.h.0.ln_2.bias.bin: [128]
transformer.h.0.mlp.c_fc.weight.bin: [512, 128]
transformer.h.0.mlp.c_fc.bias.bin: [512]
transformer.h.0.mlp.c_proj.weight.bin: [128, 512]
transformer.h.0.mlp.c_proj.bias.bin: [128]
transformer.h.1.ln_1.weight.bin: [128]
transformer.h.1.ln_1.bias.bin: [128]
transformer.h.1.attn.c_attn.weight.bin: [384, 128]
transformer.h.1.attn.c_attn.bias.bin: [384]
transformer.h.1.attn.c_proj.weight.bin: [128, 128]
transformer.h.1.attn.c_proj.bias.bin: [128]
transformer.h.1.ln_2.weight.bin: [128]
transformer.h.1.ln_2.bias.bin: [128]
transformer.h.1.mlp.c_fc.weight.bin: [512, 128]
transformer.h.1.mlp.c_fc.bias.bin: [512]
transformer.h.1.mlp.c_proj.weight.bin: [128, 512]
transformer.h.1.mlp.c_proj.bias.bin: [128]
transformer.ln_f.weight.bin: [128]
transformer.ln_f.bias.bin: [128]
lm_head.weight.bin: [65, 128]
```

I've included a export script for PyTorch models. Quite simply, you must use the model.state_dict() and export into individual files. If you want to export pre-trained GPT models, you'll need to slightly format the parameters to work correctly.

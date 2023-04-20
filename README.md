# WebGPT

![webGPT](other/misc/header.png)

After six years of development, WebGPU is about to launch across most major web browsers. This is massive: web applications now have near-native access to the GPU, with the added capacity of compute shaders.

WebGPT is a vanilla JS and HTML implementation of a transformer model, intended as a proof-of-concept as well as educational resource. WebGPT has been tested to be working with models up to 500 M parameters, though could likely support far more with further testing/optimization.

At the moment, WebGPT averages ~300ms per token on GPT-2 124M running on a 2020 M1 Mac with Chrome Canary. This could be 500% faster, if not more, with proper optimization of the kernels, buffers, the WebGPU interface. WebGPU should also receive significant speed increases as it matures.

## Running WebGPT

Running WebGPT is remarkably simple, as it's just a set of HTML + JS files. Since WebGPU is still in the process of being released, you'll need to open with a compatible browser. WebGPU is currently available on Chrome v113 but the most straightforward way to insure proper functionality is to install Chrome Canary and enable "Unsafe WebGPU" in settings.

I've included two different models: a toy GPT-Shakespeare model (which is severly undertrained haha) and GPT-2 117M. See main.js for more information on how to run these models. If you want to import custom models, take a look at misc/conversion_scripts.

## Acknowledgements

When I started this project I had no idea how transformers worked or how to implement them (or GPUs or matmul kernels or WebGPU or tokenization for that matter), so Andrej Karpathy's series on neural networks and building GPT from scratch were invaluable: [Andrej's Youtube](https://www.youtube.com/@AndrejKarpathy). I've also used some code as well from the nanoGPT repository: [nanoGPT](https://github.com/karpathy/nanoGPT).

I copied from LatitudeGames' implementation of OpenAI's GPT-3 tokenizer in Javascript: [GPT-3-Encoder](https://github.com/latitudegames/GPT-3-Encoder).

## Note: I'm looking for work!

I'm currently working on switching into working in the AI field. I'm specifically looking for opportunites at larger research labs in a variety of jobs, with the goal of breaking into the space and finding an area in which to specialize. If you're interested, check out my personal website: [Personal Website](https://depue.design/)

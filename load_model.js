/*

{
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "block_size": 64,
    "bias": false,
    "vocab_size": 65,
    "dropout": 0
}

*/

async function loadModel(filename) {
  // Load the model from json file
  var model = await (await fetch(`models/${filename}.json`)).json();
  console.log(model);

  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  const { block_size: context_size, vocab_size, n_embd, n_head: n_heads, n_layer: n_layers, bias: biasEnabled } = model.params;
  console.log("context_size", context_size, "vocab_size", vocab_size, "n_embd", n_embd, "n_heads", n_heads, "n_layers", n_layers);

  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const seq_length = 24;
  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  const embeddings = model["transformer.wte.weight"].values.flat();
  const embdBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdBuffer, 0, new Float32Array(embeddings));

  const posEmbeddings = model["transformer.wpe.weight"].values.flat();
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, new Float32Array(posEmbeddings));

  const layer_buffers = [];

  for (let i = 0; i < n_layers; i++) {
    const buffers = [];
    const prefix = `transformer.h.${i}.`;

    const layerNormAttentionGamma = model[`${prefix}ln_1.weight`].values.flat();
    const layerNormAttentionBeta = biasEnabled ? model[`${prefix}ln_1.bias`].values.flat() : new Array(n_embd).fill(0);
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, new Float32Array(layerNormAttentionGamma));
    queue.writeBuffer(normAttentionBetaBuffer, 0, new Float32Array(layerNormAttentionBeta));
    buffers.push(normAttentionGammaBuffer, normAttentionBetaBuffer);

    const qkv_weights = model[`${prefix}attn.c_attn.weight`].values.flat();
    const qkv_bias = biasEnabled ? model[`${prefix}attn.c_attn.bias`].values.flat() : new Array(3 * n_embd).fill(0);
    const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(qkvWeightsBuffer, 0, new Float32Array(qkv_weights));
    queue.writeBuffer(qkvBiasBuffer, 0, new Float32Array(qkv_bias));
    buffers.push(qkvWeightsBuffer, qkvBiasBuffer);

    const linear_weights = model[`${prefix}attn.c_proj.weight`].values.flat();
    const linear_bias = biasEnabled ? model[`${prefix}attn.c_proj.bias`].values.flat() : new Array(n_embd).fill(0);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, new Float32Array(linear_weights));
    queue.writeBuffer(linearBiasBuffer, 0, new Float32Array(linear_bias));
    buffers.push(linearWeightsBuffer, linearBiasBuffer);

    const layerNormLinearGamma = model[`${prefix}ln_2.weight`].values.flat();
    const layerNormLinearBeta = biasEnabled ? model[`${prefix}ln_2.bias`].values.flat() : new Array(n_embd).fill(0);
    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, new Float32Array(layerNormLinearGamma));
    queue.writeBuffer(normLinearBetaBuffer, 0, new Float32Array(layerNormLinearBeta));
    buffers.push(normLinearGammaBuffer, normLinearBetaBuffer);

    const firstLayerWeights = model[`${prefix}mlp.c_fc.weight`].values.flat();
    const firstLayerBias = biasEnabled ? model[`${prefix}mlp.c_fc.bias`].values.flat() : new Array(hidden_size).fill(0);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, new Float32Array(firstLayerWeights));
    queue.writeBuffer(firstLayerBiasBuffer, 0, new Float32Array(firstLayerBias));
    buffers.push(firstLayerWeightsBuffer, firstLayerBiasBuffer);

    const secondLayerWeights = model[`${prefix}mlp.c_proj.weight`].values.flat();
    const secondLayerBias = biasEnabled ? model[`${prefix}mlp.c_proj.bias`].values.flat() : new Array(n_embd).fill(0);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, new Float32Array(secondLayerWeights));
    queue.writeBuffer(secondLayerBiasBuffer, 0, new Float32Array(secondLayerBias));
    buffers.push(secondLayerWeightsBuffer, secondLayerBiasBuffer);

    layer_buffers.push(buffers);
  }

  const layerNormGamma = model["transformer.ln_f.weight"].values;
  const layerNormBeta = biasEnabled ? model["transformer.ln_f.bias"].values : new Array(n_embd).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, new Float32Array(layerNormGamma));
  queue.writeBuffer(normBetaBuffer, 0, new Float32Array(layerNormBeta));

  const deEmbeddings = model["lm_head.weight"].values.flat();
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, new Float32Array(deEmbeddings));

  return {
    device,
    queue,
    embdBuffer,
    posEmbdBuffer,
    layer_buffers,
    normGammaBuffer,
    normBetaBuffer,
    deEmbedBuffer,
  };
}

let modelParams = null;

(async () => {
  modelParams = await loadModel("bad_shakespeare");
})();

/*

const { device, queue } = await initializeWebGPU();
  const context_size = 1024;
  const seq_length = 24;
  const vocab_size = 50304;
  const n_embd = 768 / 2;
  const n_heads = 4;
  const n_layers = 12;
  const inputMatrix = new Float32Array(seq_length * vocab_size).fill(1);
  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  const inputBuffer = createBuffer(device, bufferSizeCalc(seq_length, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, inputMatrix);

  const embeddings = new Float32Array(vocab_size * n_embd).fill(-1);
  const embdBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdBuffer, 0, embeddings);

  const posEmbeddings = new Float32Array(context_size * n_embd).fill(-1);
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

  // Transformer Block Weights
  const layerNormAttentionGamma = new Array(n_embd).fill(1);
  const layerNormAttentionBeta = new Array(n_embd).fill(1);
  const qkv_bias = new Float32Array(n_embd * 3);
  const qkv_weights = new Float32Array(n_embd * 3 * n_embd);
  for (let y = 0; y < n_embd; y++) {
    for (let x = 0; x < n_embd * 3; x++) {
      qkv_bias[x] = 0.1;
      qkv_weights[y * n_embd * 3 + x] = 0.1;
    }
  }
  const linear_bias = new Float32Array(n_embd).fill(0);
  const linear_weights = new Float32Array(n_embd * n_embd);
  for (let y = 0; y < n_embd; y++) {
    for (let x = 0; x < n_embd; x++) {
      if (x === y) linear_weights[y * n_embd + x] = 0.1;
      else linear_weights[y * n_embd + x] = 0;
    }
  }
  const layerNormLinearGamma = new Float32Array(n_embd).fill(1);
  const layerNormLinearBeta = new Float32Array(n_embd).fill(0);
  const firstLayerWeights = new Float32Array(hidden_size * n_embd).fill(1);
  const firstLayerBias = new Float32Array(hidden_size).fill(1);
  const secondLayerWeights = new Float32Array(hidden_size * n_embd).fill(1);
  const secondLayerBias = new Float32Array(hidden_size).fill(1);

  const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normAttentionGammaBuffer, 0, new Float32Array(layerNormAttentionGamma));
  queue.writeBuffer(normAttentionBetaBuffer, 0, new Float32Array(layerNormAttentionBeta));

  const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  queue.writeBuffer(qkvWeightsBuffer, 0, qkv_weights);
  queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

  const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  queue.writeBuffer(linearWeightsBuffer, 0, linear_weights);
  queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

  const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
  queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);

  const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(firstLayerWeightsBuffer, 0, firstLayerWeights);
  queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);

  const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(secondLayerWeightsBuffer, 0, secondLayerWeights);
  queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);

  const layerNormGamma = new Float32Array(seq_length).fill(1);
  const layerNormBeta = new Float32Array(seq_length).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(seq_length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(seq_length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
  queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

  const deEmbeddings = new Float32Array(n_embd * vocab_size).fill(-1);
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, deEmbeddings);

  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  const startTime = performance.now();
  const result = await runGPT(
    device,
    queue,
    seq_length,
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    attentionDotProductScale,
    bufferSizeCalc,
    inputBuffer,
    embdBuffer,
    posEmbdBuffer,
    normAttentionGammaBuffer,
    normAttentionBetaBuffer,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer,
    normLinearGammaBuffer,
    normLinearBetaBuffer,
    firstLayerWeightsBuffer,
    firstLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerBiasBuffer,
    normGammaBuffer,
    normBetaBuffer,
    deEmbedBuffer
  );
  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);

  printMatrix(seq_length, vocab_size, new Float32Array(result));

  */

/*


  transformer.wte.weight: torch.Size([65, 128])
transformer.wpe.weight: torch.Size([64, 128])
transformer.h.0.ln_1.weight: torch.Size([128])
transformer.h.0.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.0.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.0.ln_2.weight: torch.Size([128])
transformer.h.0.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.0.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.1.ln_1.weight: torch.Size([128])
transformer.h.1.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.1.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.1.ln_2.weight: torch.Size([128])
transformer.h.1.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.1.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.2.ln_1.weight: torch.Size([128])
transformer.h.2.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.2.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.2.ln_2.weight: torch.Size([128])
transformer.h.2.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.2.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.3.ln_1.weight: torch.Size([128])
transformer.h.3.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.3.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.3.ln_2.weight: torch.Size([128])
transformer.h.3.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.3.mlp.c_proj.weight: torch.Size([128, 512])
transformer.ln_f.weight: torch.Size([128])
lm_head.weight: torch.Size([65, 128])

With bias:

transformer.wte.weight: torch.Size([65, 64])
transformer.wpe.weight: torch.Size([64, 64])

transformer.h.0.ln_1.weight: torch.Size([64])
transformer.h.0.ln_1.bias: torch.Size([64])
transformer.h.0.attn.c_attn.weight: torch.Size([192, 64])
transformer.h.0.attn.c_attn.bias: torch.Size([192])
transformer.h.0.attn.c_proj.weight: torch.Size([64, 64])
transformer.h.0.attn.c_proj.bias: torch.Size([64])
transformer.h.0.ln_2.weight: torch.Size([64])
transformer.h.0.ln_2.bias: torch.Size([64])
transformer.h.0.mlp.c_fc.weight: torch.Size([256, 64])
transformer.h.0.mlp.c_fc.bias: torch.Size([256])
transformer.h.0.mlp.c_proj.weight: torch.Size([64, 256])
transformer.h.0.mlp.c_proj.bias: torch.Size([64])

transformer.h.1.ln_1.weight: torch.Size([64])
transformer.h.1.ln_1.bias: torch.Size([64])
transformer.h.1.attn.c_attn.weight: torch.Size([192, 64])
transformer.h.1.attn.c_attn.bias: torch.Size([192])
transformer.h.1.attn.c_proj.weight: torch.Size([64, 64])
transformer.h.1.attn.c_proj.bias: torch.Size([64])
transformer.h.1.ln_2.weight: torch.Size([64])
transformer.h.1.ln_2.bias: torch.Size([64])
transformer.h.1.mlp.c_fc.weight: torch.Size([256, 64])
transformer.h.1.mlp.c_fc.bias: torch.Size([256])
transformer.h.1.mlp.c_proj.weight: torch.Size([64, 256])
transformer.h.1.mlp.c_proj.bias: torch.Size([64])

transformer.ln_f.weight: torch.Size([64])
transformer.ln_f.bias: torch.Size([64])
lm_head.weight: torch.Size([65, 64])

    */

async function loadModel(filename) {
  console.log("Loading model:", filename);
  // Load the model from json file
  var model = await (await fetch(`models/${filename}.json`)).json();
  rawModel = model;

  console.log("Model loaded:", model);

  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  const { block_size: context_size, vocab_size, n_embd, n_head: n_heads, n_layer: n_layers, bias: biasEnabled } = model.params;
  console.log("context_size", context_size, "vocab_size", vocab_size, "n_embd", n_embd, "n_heads", n_heads, "n_layers", n_layers, "biasEnabled", biasEnabled);

  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  const embeddings = model["transformer.wte.weight"].values.flat().map(parseFloat);
  const embdBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdBuffer, 0, new Float32Array(embeddings));

  const posEmbeddings = model["transformer.wpe.weight"].values.flat().map(parseFloat);
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, new Float32Array(posEmbeddings));

  const layer_buffers = [];

  for (let i = 0; i < n_layers; i++) {
    const buffers = [];
    const prefix = `transformer.h.${i}.`;

    const layerNormAttentionGamma = model[`${prefix}ln_1.weight`].values.flat().map(parseFloat);
    const layerNormAttentionBeta = biasEnabled ? model[`${prefix}ln_1.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, new Float32Array(layerNormAttentionGamma));
    queue.writeBuffer(normAttentionBetaBuffer, 0, new Float32Array(layerNormAttentionBeta));
    buffers.push(normAttentionGammaBuffer, normAttentionBetaBuffer);

    const qkv_weights = model[`${prefix}attn.c_attn.weight`].values.flat().map(parseFloat);
    const qkv_bias = biasEnabled ? model[`${prefix}attn.c_attn.bias`].values.flat().map(parseFloat) : new Array(3 * n_embd).fill(0);
    const qkvWeightsBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, 3 * n_embd),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    queue.writeBuffer(qkvWeightsBuffer, 0, new Float32Array(transposeArray(qkv_weights, 3 * n_embd, n_embd)));
    queue.writeBuffer(qkvBiasBuffer, 0, new Float32Array(qkv_bias));
    buffers.push(qkvWeightsBuffer, qkvBiasBuffer);

    const linear_weights = model[`${prefix}attn.c_proj.weight`].values.flat().map(parseFloat);
    const linear_bias = biasEnabled ? model[`${prefix}attn.c_proj.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, new Float32Array(transposeArray(linear_weights, n_embd, n_embd)));
    queue.writeBuffer(linearBiasBuffer, 0, new Float32Array(linear_bias));
    buffers.push(linearWeightsBuffer, linearBiasBuffer);

    const layerNormLinearGamma = model[`${prefix}ln_2.weight`].values.flat().map(parseFloat);
    const layerNormLinearBeta = biasEnabled ? model[`${prefix}ln_2.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, new Float32Array(layerNormLinearGamma));
    queue.writeBuffer(normLinearBetaBuffer, 0, new Float32Array(layerNormLinearBeta));
    buffers.push(normLinearGammaBuffer, normLinearBetaBuffer);

    const firstLayerWeights = model[`${prefix}mlp.c_fc.weight`].values.flat().map(parseFloat);
    const firstLayerBias = biasEnabled ? model[`${prefix}mlp.c_fc.bias`].values.flat().map(parseFloat) : new Array(hidden_size).fill(0);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, new Float32Array(transposeArray(firstLayerWeights, hidden_size, n_embd)));
    queue.writeBuffer(firstLayerBiasBuffer, 0, new Float32Array(firstLayerBias));
    buffers.push(firstLayerWeightsBuffer, firstLayerBiasBuffer);

    const secondLayerWeights = model[`${prefix}mlp.c_proj.weight`].values.flat().map(parseFloat);
    const secondLayerBias = biasEnabled ? model[`${prefix}mlp.c_proj.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, new Float32Array(transposeArray(secondLayerWeights, n_embd, hidden_size)));
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

  const deEmbeddings = model["lm_head.weight"].values.flat().map(parseFloat);
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, new Float32Array(transposeArray(deEmbeddings, vocab_size, n_embd)));

  // console.log(new Float32Array(transposeArray(deEmbeddings, vocab_size, n_embd)));

  return {
    device,
    queue,
    params: {
      attentionDotProductScale,
      biasEnabled,
      n_embd,
      n_heads,
      n_layers,
      vocab_size,
      hidden_size,
      context_size,
    },
    embdBuffer,
    posEmbdBuffer,
    layer_buffers,
    normGammaBuffer,
    normBetaBuffer,
    deEmbedBuffer,
  };
}

function reshapeRecursively(flatArray, shape) {
  if (shape.length === 1) {
    return flatArray.slice(0, shape[0]);
  }

  let result = [];
  let elementsPerSection = shape.slice(1).reduce((a, b) => a * b);
  for (let i = 0; i < flatArray.length; i += elementsPerSection) {
    result.push(reshapeRecursively(flatArray.slice(i, i + elementsPerSection), shape.slice(1)));
  }

  return result;
}

async function loadValidateModel(validateFile) {
  const validateData = await (await fetch(`test/${validateFile}`)).json();

  const steps = [];
  for (let i = 0; i < validateData.length; i++) {
    const loadedData = {};
    for (const key in validateData[i]) {
      const shape = validateData[i][key].shape;
      const data = validateData[i][key].data.flat(Infinity).map((value) => parseFloat(value));
      const typedArray = new Float32Array(data);

      loadedData[key] = {
        shape,
        data: reshapeRecursively(typedArray, shape),
      };
    }
    steps.push(loadedData);
  }

  return steps;
}

function transposeArray(array, input_rows, input_cols) {
  if (array.length !== input_rows * input_cols) {
    throw new Error("Transpose dims failed");
  }

  const transpose = [];
  for (let col = 0; col < input_cols; col++) {
    for (let row = 0; row < input_rows; row++) {
      transpose.push(array[row * input_cols + col]);
    }
  }
  return transpose;
}

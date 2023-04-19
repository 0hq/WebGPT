async function loadGPTModel(folder) {
  console.log("Loading model:", folder);
  // Load the model from json file
  // var model = await (await fetch(`models/${folder}.json`)).json();

  // console.log("Model loaded:", model);

  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  const paramsJSON = await (await fetch(`models/${folder}/params_gpt.json`)).json();
  const { block_size: context_size, vocab_size, n_embd, n_head: n_heads, n_layer: n_layers, bias: biasEnabled } = paramsJSON;
  console.log("context_size", context_size, "vocab_size", vocab_size, "n_embd", n_embd, "n_heads", n_heads, "n_layers", n_layers, "biasEnabled", biasEnabled);

  console.log("Device:", device.limits);

  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  const posEmbeddingsJSON = await fetch(`models/${folder}/transformer.wpe.weight_gpt.json`);
  // console.log("posEmbeddingsJSON", posEmbeddingsJSON);
  const posEmbeddingsJSON2 = await posEmbeddingsJSON.json();
  // console.log("posEmbeddingsJSON2", posEmbeddingsJSON2);
  const posEmbeddings = posEmbeddingsJSON2.values.flat().map(parseFloat);
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, new Float32Array(posEmbeddings));

  const layer_buffers = [];

  for (let i = 0; i < n_layers; i++) {
    const buffers = [];
    const prefix = `transformer.h.${i}.`;

    console.log("Loading layer", i);

    const layerNormAttentionGammaJSON = await (await fetch(`models/${folder}/${prefix}ln_1.weight_gpt.json`)).json();
    const layerNormAttentionGamma = layerNormAttentionGammaJSON.values.flat().map(parseFloat);
    const layerNormAttentionBetaJSON = await (await fetch(`models/${folder}/${prefix}ln_1.bias_gpt.json`)).json();
    const layerNormAttentionBeta = biasEnabled ? layerNormAttentionBetaJSON.values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    // console.log(layerNormAttentionBetaJSON);
    // console.log(layerNormAttentionGammaJSON);
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, new Float32Array(layerNormAttentionGamma));
    queue.writeBuffer(normAttentionBetaBuffer, 0, new Float32Array(layerNormAttentionBeta));
    buffers.push(normAttentionGammaBuffer, normAttentionBetaBuffer);

    // const qkv_weights = model[`${prefix}attn.c_attn.weight`].values.flat().map(parseFloat);
    // const qkv_bias = biasEnabled ? model[`${prefix}attn.c_attn.bias`].values.flat().map(parseFloat) : new Array(3 * n_embd).fill(0);

    const qkv_weightsJSON = await (await fetch(`models/${folder}/${prefix}attn.c_attn.weight_gpt.json`)).json();
    const qkv_weights = qkv_weightsJSON.values.flat().map(parseFloat);
    const qkv_biasJSON = await (await fetch(`models/${folder}/${prefix}attn.c_attn.bias_gpt.json`)).json();
    const qkv_bias = biasEnabled ? qkv_biasJSON.values.flat().map(parseFloat) : new Array(3 * n_embd).fill(0);
    // console.log("qkv_bias", qkv_biasJSON);
    // console.log("qkv_weights", qkv_weightsJSON);
    const qkvWeightsBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, 3 * n_embd),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    queue.writeBuffer(qkvWeightsBuffer, 0, new Float32Array(transposeArray(qkv_weights, 3 * n_embd, n_embd)));
    queue.writeBuffer(qkvBiasBuffer, 0, new Float32Array(qkv_bias));
    buffers.push(qkvWeightsBuffer, qkvBiasBuffer);

    // const linear_weights = model[`${prefix}attn.c_proj.weight`].values.flat().map(parseFloat);
    // const linear_bias = biasEnabled ? model[`${prefix}attn.c_proj.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

    const linear_weightsJSON = await (await fetch(`models/${folder}/${prefix}attn.c_proj.weight_gpt.json`)).json();
    const linear_weights = linear_weightsJSON.values.flat().map(parseFloat);
    const linear_biasJSON = await (await fetch(`models/${folder}/${prefix}attn.c_proj.bias_gpt.json`)).json();
    const linear_bias = biasEnabled ? linear_biasJSON.values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    // console.log("linear_bias", linear_biasJSON);
    // console.log("linear_weights", linear_weightsJSON);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, new Float32Array(transposeArray(linear_weights, n_embd, n_embd)));
    queue.writeBuffer(linearBiasBuffer, 0, new Float32Array(linear_bias));
    buffers.push(linearWeightsBuffer, linearBiasBuffer);

    // const layerNormLinearGamma = model[`${prefix}ln_2.weight`].values.flat().map(parseFloat);
    // const layerNormLinearBeta = biasEnabled ? model[`${prefix}ln_2.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

    const layerNormLinearGammaJSON = await (await fetch(`models/${folder}/${prefix}ln_2.weight_gpt.json`)).json();
    const layerNormLinearGamma = layerNormLinearGammaJSON.values.flat().map(parseFloat);
    const layerNormLinearBetaJSON = await (await fetch(`models/${folder}/${prefix}ln_2.bias_gpt.json`)).json();
    const layerNormLinearBeta = biasEnabled ? layerNormLinearBetaJSON.values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    // console.log("beta layer norm", layerNormLinearBetaJSON);
    // console.log("gamma layer norm", layerNormLinearGammaJSON);

    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, new Float32Array(layerNormLinearGamma));
    queue.writeBuffer(normLinearBetaBuffer, 0, new Float32Array(layerNormLinearBeta));
    buffers.push(normLinearGammaBuffer, normLinearBetaBuffer);

    // const firstLayerWeights = model[`${prefix}mlp.c_fc.weight`].values.flat().map(parseFloat);
    // const firstLayerBias = biasEnabled ? model[`${prefix}mlp.c_fc.bias`].values.flat().map(parseFloat) : new Array(hidden_size).fill(0);

    const firstLayerWeightsJSON = await (await fetch(`models/${folder}/${prefix}mlp.c_fc.weight_gpt.json`)).json();
    const firstLayerWeights = firstLayerWeightsJSON.values.flat().map(parseFloat);
    const firstLayerBiasJSON = await (await fetch(`models/${folder}/${prefix}mlp.c_fc.bias_gpt.json`)).json();
    const firstLayerBias = biasEnabled ? firstLayerBiasJSON.values.flat().map(parseFloat) : new Array(hidden_size).fill(0);
    // console.log("first layer bias", firstLayerBiasJSON);
    // console.log("first layer weights", firstLayerWeightsJSON);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, new Float32Array(transposeArray(firstLayerWeights, hidden_size, n_embd)));
    queue.writeBuffer(firstLayerBiasBuffer, 0, new Float32Array(firstLayerBias));
    buffers.push(firstLayerWeightsBuffer, firstLayerBiasBuffer);

    // const secondLayerWeights = model[`${prefix}mlp.c_proj.weight`].values.flat().map(parseFloat);
    // const secondLayerBias = biasEnabled ? model[`${prefix}mlp.c_proj.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

    const secondLayerWeightsJSON = await (await fetch(`models/${folder}/${prefix}mlp.c_proj.weight_gpt.json`)).json();
    const secondLayerWeights = secondLayerWeightsJSON.values.flat().map(parseFloat);
    const secondLayerBiasJSON = await (await fetch(`models/${folder}/${prefix}mlp.c_proj.bias_gpt.json`)).json();
    const secondLayerBias = biasEnabled ? secondLayerBiasJSON.values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    // console.log("second layer bias", secondLayerBiasJSON);
    // console.log("second layer weights", secondLayerWeightsJSON);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, new Float32Array(transposeArray(secondLayerWeights, n_embd, hidden_size)));
    queue.writeBuffer(secondLayerBiasBuffer, 0, new Float32Array(secondLayerBias));
    buffers.push(secondLayerWeightsBuffer, secondLayerBiasBuffer);

    layer_buffers.push(buffers);
  }

  // const layerNormGamma = model["transformer.ln_f.weight"].values;
  // const layerNormBeta = biasEnabled ? model["transformer.ln_f.bias"].values : new Array(n_embd).fill(0);

  const layerNormGammaJSON = await (await fetch(`models/${folder}/transformer.ln_f.weight_gpt.json`)).json();
  const layerNormGamma = layerNormGammaJSON.values.flat().map(parseFloat);
  const layerNormBetaJSON = await (await fetch(`models/${folder}/transformer.ln_f.bias_gpt.json`)).json();
  const layerNormBeta = biasEnabled ? layerNormBetaJSON.values.flat().map(parseFloat) : new Array(n_embd).fill(0);
  // console.log("beta layer norm", layerNormBetaJSON);
  // console.log("gamma layer norm", layerNormGammaJSON);

  const normGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, new Float32Array(layerNormGamma));
  queue.writeBuffer(normBetaBuffer, 0, new Float32Array(layerNormBeta));

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
    embdBuffer: null,
    posEmbdBuffer,
    layer_buffers,
    normGammaBuffer,
    normBetaBuffer,
    deEmbedBuffer: null,
  };
}

async function loadFakeGPT2() {
  console.log("FakeGPT: Loading model as all zeros...");

  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  const paramsJSON = await (await fetch(`models/gpt2/params_gpt.json`)).json();
  const { block_size: context_size, vocab_size, n_embd, n_head: n_heads, n_layer: n_layers, bias: biasEnabled } = paramsJSON;
  console.log("context_size", context_size, "vocab_size", vocab_size, "n_embd", n_embd, "n_heads", n_heads, "n_layers", n_layers, "biasEnabled", biasEnabled);

  console.log("Device:", device.limits);

  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  console.log("Loading embeddings...");
  const embeddings = new Array(vocab_size * (n_embd - 400)).fill(0.1);
  console.warn("WARNING: Using fake embeddings offset by 400.");
  const embdBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd - 400), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdBuffer, 0, new Float32Array(embeddings));

  const posEmbeddings = new Array(context_size * n_embd).fill(0.1);
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, new Float32Array(posEmbeddings));

  const layer_buffers = [];

  for (let i = 0; i < n_layers; i++) {
    const buffers = [];

    console.log("Loading layer...", i);

    const layerNormAttentionGamma = new Array(n_embd).fill(0.1);
    const layerNormAttentionBeta = new Array(n_embd).fill(0.1);
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, new Float32Array(layerNormAttentionGamma));
    queue.writeBuffer(normAttentionBetaBuffer, 0, new Float32Array(layerNormAttentionBeta));
    buffers.push(normAttentionGammaBuffer, normAttentionBetaBuffer);

    const qkv_weights = new Array(n_embd * 3 * n_embd).fill(0.1);
    const qkv_bias = new Array(3 * n_embd).fill(0.1);
    const qkvWeightsBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, 3 * n_embd),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    queue.writeBuffer(qkvWeightsBuffer, 0, new Float32Array(transposeArray(qkv_weights, 3 * n_embd, n_embd)));
    queue.writeBuffer(qkvBiasBuffer, 0, new Float32Array(qkv_bias));
    buffers.push(qkvWeightsBuffer, qkvBiasBuffer);

    const linear_weights = new Array(n_embd * n_embd).fill(0.1);
    const linear_bias = new Array(n_embd).fill(0.1);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, new Float32Array(transposeArray(linear_weights, n_embd, n_embd)));
    queue.writeBuffer(linearBiasBuffer, 0, new Float32Array(linear_bias));
    buffers.push(linearWeightsBuffer, linearBiasBuffer);

    const layerNormLinearGamma = new Array(n_embd).fill(0.1);
    const layerNormLinearBeta = new Array(n_embd).fill(0.1);
    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, new Float32Array(layerNormLinearGamma));
    queue.writeBuffer(normLinearBetaBuffer, 0, new Float32Array(layerNormLinearBeta));
    buffers.push(normLinearGammaBuffer, normLinearBetaBuffer);

    const firstLayerWeights = new Array(hidden_size * n_embd).fill(0.1);
    const firstLayerBias = new Array(hidden_size).fill(0.1);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, new Float32Array(transposeArray(firstLayerWeights, hidden_size, n_embd)));
    queue.writeBuffer(firstLayerBiasBuffer, 0, new Float32Array(firstLayerBias));
    buffers.push(firstLayerWeightsBuffer, firstLayerBiasBuffer);

    const secondLayerWeights = new Array(n_embd * hidden_size).fill(0.1);
    const secondLayerBias = new Array(n_embd).fill(0.1);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, new Float32Array(transposeArray(secondLayerWeights, n_embd, hidden_size)));
    queue.writeBuffer(secondLayerBiasBuffer, 0, new Float32Array(secondLayerBias));
    buffers.push(secondLayerWeightsBuffer, secondLayerBiasBuffer);

    layer_buffers.push(buffers);
  }

  const layerNormGamma = new Array(n_embd).fill(0.1);
  const layerNormBeta = new Array(n_embd).fill(0.1);

  const normGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, new Float32Array(layerNormGamma));
  queue.writeBuffer(normBetaBuffer, 0, new Float32Array(layerNormBeta));

  console.log("Loading de-embeddings...");
  const deEmbeddings = new Array(vocab_size * (n_embd - 400)).fill(0.1);
  console.warn("WARNING: Using fake embeddings offset by 400.");
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd - 400, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, new Float32Array(transposeArray(deEmbeddings, vocab_size, n_embd)));

  console.log("Model loaded.");
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

function loadFloat32ArrayFromJSONOboe(url) {
  return new Promise((resolve, reject) => {
    const float32List = new Float32Array(38597376);
    const index = 0;

    oboe(url)
      .node("!values.*", function (value) {
        float32List[index] = parseFloat(value);
      })
      .done(function () {
        const float32Array = float32List;
        console.log("Float32Array:", float32Array);
        resolve(float32Array);
      })
      .fail(function (error) {
        console.error("An error occurred:", error);
        reject(error);
      });
  });
}

async function loadFloat32ArrayFromJSON(url) {
  try {
    // Fetch the JSON file
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Error loading ${url}: ${response.statusText}`);
    }

    // Parse the JSON file to retrieve the array of float32s
    const float32List = (await response.json()).values;
    console.log("float32List:", float32List);

    // Create a new Float32Array and fill it with the parsed float32 values
    const float32Array = new Float32Array(float32List.length);

    for (let i = 0; i < float32List.length; i++) {
      float32Array[i] = float32List[i];
    }

    console.log("Float32Array:", float32Array);
    return float32Array;
  } catch (error) {
    console.error("An error occurred:", error);
  }
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
    console.error("Transpose dims failed, not transposing!");
    return array;
    // throw new Error("Transpose dims failed");
  }

  const transpose = [];
  for (let col = 0; col < input_cols; col++) {
    for (let row = 0; row < input_rows; row++) {
      transpose.push(array[row * input_cols + col]);
    }
  }
  return transpose;
}

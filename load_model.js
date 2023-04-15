async function loadModel(filename) {
  console.log("Loading model:", filename);
  // Load the model from json file
  var model = await (await fetch(`models/${filename}.json`)).json();

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
    const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(qkvWeightsBuffer, 0, new Float32Array(qkv_weights));
    queue.writeBuffer(qkvBiasBuffer, 0, new Float32Array(qkv_bias));
    buffers.push(qkvWeightsBuffer, qkvBiasBuffer);

    const linear_weights = model[`${prefix}attn.c_proj.weight`].values.flat().map(parseFloat);
    const linear_bias = biasEnabled ? model[`${prefix}attn.c_proj.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, new Float32Array(linear_weights));
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
    queue.writeBuffer(firstLayerWeightsBuffer, 0, new Float32Array(firstLayerWeights));
    queue.writeBuffer(firstLayerBiasBuffer, 0, new Float32Array(firstLayerBias));
    buffers.push(firstLayerWeightsBuffer, firstLayerBiasBuffer);

    const secondLayerWeights = model[`${prefix}mlp.c_proj.weight`].values.flat().map(parseFloat);
    const secondLayerBias = biasEnabled ? model[`${prefix}mlp.c_proj.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);
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

  const deEmbeddings = model["lm_head.weight"].values.flat().map(parseFloat);
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, new Float32Array(deEmbeddings));

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

async function runInference(prompt) {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, embdBuffer, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, deEmbedBuffer } = modelParams;
  const { attentionDotProductScale, biasEnabled, n_embd, n_heads, n_layers, vocab_size, hidden_size, context_size } = params;

  const seq_length = prompt.length;
  const inputMatrix = new Float32Array(seq_length * vocab_size);
  for (let i = 0; i < seq_length; i++) {
    inputMatrix[i * vocab_size + prompt[i]] = 1;
  }
  // printMatrix(seq_length, vocab_size, new Float32Array(inputMatrix));
  const inputBuffer = createBuffer(device, bufferSizeCalc(seq_length, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, inputMatrix);

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
    inputBuffer,
    embdBuffer,
    posEmbdBuffer,
    layer_buffers,
    normGammaBuffer,
    normBetaBuffer,
    deEmbedBuffer
  );
  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);

  // printMatrix(seq_length, vocab_size, new Float32Array(result));

  return new Float32Array(result);
}

async function runGPT(
  device,
  queue,
  seq_length,
  vocab_size,
  n_embd,
  n_heads,
  n_layers,
  attentionDotProductScale,
  inputBuffer,
  embdBuffer,
  posEmbdBuffer,
  layer_buffers,
  normGammaBuffer,
  normBetaBuffer,
  deEmbedBuffer
) {
  const commandEncoder = device.createCommandEncoder();

  const embdOutputBuffer = inlineMatMul(device, queue, commandEncoder, inputBuffer, embdBuffer, seq_length, n_embd, vocab_size);
  // Crop the position embeddings to the correct size.
  const posEmbdOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(seq_length, n_embd),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  commandEncoder.copyBufferToBuffer(
    posEmbdBuffer, // Source buffer (original position embeddings)
    0, // Source offset (starting from the beginning of the buffer)
    posEmbdOutputBuffer, // Destination buffer (cropped buffer)
    0, // Destination offset (starting from the beginning of the cropped buffer)
    bufferSizeCalc(seq_length, n_embd) // Number of bytes to copy
  );
  // Residual connection is just elementwise addition, can be used for combining embedding and position embedding.
  const embeddedInputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, embdOutputBuffer, posEmbdOutputBuffer);
  let layerBuffer = embeddedInputBuffer;

  // Used for validation.
  const buffers = [];

  for (let i = 0; i < n_layers; i++) {
    const layer_params = layer_buffers[i];
    const {
      layerNormAttentionOutputBuffer,
      attentionOutputBuffer,
      residualAttentionOutputBuffer,
      layerNormLinearOutputBuffer,
      linearOutputBuffer,
      residualLinearOutputBuffer,
    } = transformerBlock(device, queue, commandEncoder, seq_length, n_embd, n_heads, attentionDotProductScale, layerBuffer, ...layer_params);
    buffers.push({
      layerNormAttentionOutputBuffer,
      attentionOutputBuffer,
      residualAttentionOutputBuffer,
      layerNormLinearOutputBuffer,
      linearOutputBuffer,
      residualLinearOutputBuffer,
    });
    layerBuffer = residualLinearOutputBuffer;
  }

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerBuffer, normGammaBuffer, normBetaBuffer);

  const deEmbedOutputBuffer = inlineMatMul(device, queue, commandEncoder, layerNormOutputBuffer, deEmbedBuffer, seq_length, vocab_size, n_embd);

  // OUTPUT and VALIDATION

  const outputEmbedBuffer = createOutputBuffer(device, commandEncoder, embeddedInputBuffer, seq_length, n_embd);
  const outputBlockBuffers = [];
  for (let i = 0; i < n_layers; i++) {
    const block = buffers[i];
    const outputLayerNormAttentionBuffer = createOutputBuffer(device, commandEncoder, block.layerNormAttentionOutputBuffer, seq_length, n_embd);
    const outputAttentionBuffer = createOutputBuffer(device, commandEncoder, block.attentionOutputBuffer, seq_length, n_embd);
    const outputResidualAttentionBuffer = createOutputBuffer(device, commandEncoder, block.residualAttentionOutputBuffer, seq_length, n_embd);
    const outputLayerNormLinearBuffer = createOutputBuffer(device, commandEncoder, block.layerNormLinearOutputBuffer, seq_length, n_embd);
    const outputLinearBuffer = createOutputBuffer(device, commandEncoder, block.linearOutputBuffer, seq_length, n_embd);
    const outputResidualLinearBuffer = createOutputBuffer(device, commandEncoder, block.residualLinearOutputBuffer, seq_length, n_embd);
    outputBlockBuffers.push([
      outputLayerNormAttentionBuffer,
      outputAttentionBuffer,
      outputResidualAttentionBuffer,
      outputLayerNormLinearBuffer,
      outputLinearBuffer,
      outputResidualLinearBuffer,
    ]);
  }
  const outputLayerBuffer = createOutputBuffer(device, commandEncoder, layerBuffer, seq_length, n_embd);
  const outputLayerNormBuffer = createOutputBuffer(device, commandEncoder, layerNormOutputBuffer, seq_length, n_embd);
  const outputBuffer = createOutputBuffer(device, commandEncoder, deEmbedOutputBuffer, seq_length, vocab_size);

  queue.submit([commandEncoder.finish()]);

  await outputEmbedBuffer.mapAsync(GPUMapMode.READ);
  for (let i = 0; i < n_layers; i++) {
    const block = outputBlockBuffers[i];
    for (let j = 0; j < block.length; j++) {
      await block[j].mapAsync(GPUMapMode.READ);
    }
  }
  await outputLayerBuffer.mapAsync(GPUMapMode.READ);
  await outputLayerNormBuffer.mapAsync(GPUMapMode.READ);
  await outputBuffer.mapAsync(GPUMapMode.READ);

  // const outputEmbedBufferMat = formatAsMatrix(, seq_length, n_embd);
  // const outputBlockBuffersMat = [];
  // for (let i = 0; i < n_layers; i++) {
  //   outputBlockBuffersMat.push(formatAsMatrix(new Float32Array(outputBlockBuffers[i].getMappedRange()), seq_length, n_embd));
  // }
  // const outputLayerBufferMat = formatAsMatrix(new Float32Array(outputLayerBuffer.getMappedRange()), seq_length, n_embd);
  // const outputLayerNormBufferMat = formatAsMatrix(new Float32Array(outputLayerNormBuffer.getMappedRange()), seq_length, n_embd);

  // You can't read twice from mapped range.
  const output = outputBuffer.getMappedRange();

  console.log("Validating output...");
  console.log("Validating embedding...");
  // console.log(new Float32Array(outputEmbedBuffer.getMappedRange()), validateModel[tokenIndex].tok_pos_emb);
  validateResult(new Float32Array(outputEmbedBuffer.getMappedRange()), validateModel[tokenIndex].tok_pos_emb);
  console.log("Validating blocks...");
  for (let i = 0; i < n_layers; i++) {
    console.log(`\tValidating block ${i}...`);
    const block = outputBlockBuffers[i];
    console.log("\t\tValidating first layer norm...");
    validateResult(new Float32Array(outputBlockBuffers[i][0].getMappedRange()), validateModel[tokenIndex][`block${i}_ln1`]);
    console.log("\t\tValidating attention...");
    validateResult(new Float32Array(outputBlockBuffers[i][1].getMappedRange()), validateModel[tokenIndex][`block${i}_attn`]);
    console.log("\t\tValidating residual attention...");
    validateResult(new Float32Array(outputBlockBuffers[i][2].getMappedRange()), validateModel[tokenIndex][`block${i}_r1`]);
    console.log("\t\tValidating second layer norm...");
    validateResult(new Float32Array(outputBlockBuffers[i][3].getMappedRange()), validateModel[tokenIndex][`block${i}_ln2`]);
    console.log("\t\tValidating mlp...");
    validateResult(new Float32Array(outputBlockBuffers[i][4].getMappedRange()), validateModel[tokenIndex][`block${i}_mlp`]);
    console.log("\t\tValidating residual mlp...");
    validateResult(new Float32Array(outputBlockBuffers[i][5].getMappedRange()), validateModel[tokenIndex][`block${i}_r2`]);
  }
  console.log("Validating layer norm...");
  validateResult(new Float32Array(outputLayerNormBuffer.getMappedRange()), validateModel[tokenIndex].ln_f);
  console.log("Validating logits...");
  validateResult(new Float32Array(output), validateModel[tokenIndex].logits);

  return output;
}

function createOutputBuffer(device, commandEncoder, buffer, rows, cols) {
  const outputBufferSize = bufferSizeCalc(rows, cols);
  const outputBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  commandEncoder.copyBufferToBuffer(buffer, 0, outputBuffer, 0, outputBufferSize);
  return outputBuffer;
}

let tokenIndex = 0;
async function generateFromModel(prompt, max_new_tokens) {
  if (!modelParams || !stoi || !itos) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("Starting generation with prompt", prompt);
  prompt = prompt.split("").map((c) => stoi[c]);
  console.log("Parsed prompt", prompt);

  const context_size = modelParams.params.context_size;
  console.log("block_size", context_size);
  for (let i = 0; i < max_new_tokens; i++) {
    tokenIndex = i;

    // console.log("prompt", prompt);
    const idx_cond = prompt.slice(-context_size);
    // console.log("running inference on sequence", idx_cond);
    const logits = await runInference(idx_cond);

    // pluck the logits at the final step and scale by desired temperature
    // const logits_scaled = logits; // / temperature;

    // apply softmax to convert logits to (normalized) probabilities
    const probs = simpleSoftmax(logits);

    console.log("Probs", probs);
    console.log("Max prob:", Math.max(...probs));
    console.log("Max prob index:", probs.indexOf(Math.max(...probs)), "char:", itos[probs.indexOf(Math.max(...probs))]);

    // sample from the distribution
    const idx_next = sampleFromDistribution(probs);
    // append sampled index to the running sequence and continue
    // console.log("generated", idx_next);
    prompt = prompt.concat(idx_next);
  }

  console.log("Output ints:", prompt);
  const text = prompt.map((i) => itos[i]).join("");
  console.log("Output:", text);
}

let itos = null;
let stoi = null;
let modelParams = null;
let bufferSizeCalc = null;

let validateModel = null;
const validateFile = "generation.json";

(async () => {
  modelParams = await loadModel("state_dict");
  console.log("Params:", modelParams);

  const tokenDict = await (await fetch("models/tokens.json")).json();

  itos = tokenDict.itos;
  stoi = tokenDict.stoi;

  console.log("Tokens:", tokenDict);
  console.log("Unique Tokens:", new Set(Object.values(tokenDict.itos)));

  console.log("Model finished loading.");

  const validateData = await loadValidateModel(validateFile);

  validateModel = validateData;

  console.log("Validate model loaded", validateData);

  generateFromModel("What is the answer to life, the universe, and everything?", 1);
})();

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

function validateResult(result, validate) {
  const resultArray = formatAsMatrix(result, validate.shape[1], validate.shape[2]);
  const validateArray = validate.data[0]; // Unpack from batch of 1

  const equal = checkEqualMatrices(resultArray, validateArray);

  if (!equal) {
    // console.log("Result:", result);
    // console.log("Validate:", validate);
    console.log("Result mat:", resultArray);
    console.log("Validate mat:", validateArray);

    // Subtract the matrices
    const diff = subtractMatrices(resultArray, validateArray);
    console.log("Diff mat:", diff);

    throw new Error("Test failed");
  } else {
    console.log("Test passed!");
  }
}

function subtractMatrices(a, b) {
  const result = [];
  for (let i = 0; i < a.length; i++) {
    result.push([]);
    for (let j = 0; j < a[i].length; j++) {
      result[i].push(a[i][j] - b[i][j]);
    }
  }

  return result;
}

let tokenIndex = 0;

let rawModel = null;

let itos = null;
let stoi = null;
let modelParams = null;
let bufferSizeCalc = null;

let validateModel = null;
const validateFile = "generation.json";
const doValidation = true;

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

  // runAttentionTest();
})();

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

    console.log("Logits", logits);

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

async function runAttentionTest() {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, embdBuffer, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, deEmbedBuffer } = modelParams;
  const { attentionDotProductScale, biasEnabled, n_embd, n_heads, n_layers, vocab_size, hidden_size, context_size } = params;
  const commandEncoder = device.createCommandEncoder();

  const seq_length = 57;

  const layerNormAttention = validateModel[tokenIndex][`block0_ln1`].data[0];
  const layerNormAttentionOutput = [];
  for (let i = 0; i < seq_length; i++) {
    layerNormAttentionOutput.push(...layerNormAttention[i]);
  }

  console.log("input", layerNormAttention);

  const layerNormAttentionOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(layerNormAttentionOutputBuffer, 0, new Float32Array(layerNormAttentionOutput));

  const startTime = performance.now();

  const layer_buffer = layer_buffers[0];

  // const {
  //   qkvResultBuffer,
  //   splitQResultBuffer,
  //   splitKResultBuffer,
  //   splitVResultBuffer,
  //   attentionWeightsResultBuffer,
  //   multiplyResultBuffer,
  //   causalMaskResultBuffer,
  //   attentionValuesResultBuffer,
  //   linearResultBuffer,
  // } = devInlineAttention(
  //   device,
  //   queue,
  //   commandEncoder,
  //   seq_length,
  //   n_embd,
  //   attentionDotProductScale,
  //   layerNormAttentionOutputBuffer,
  //   n_heads,
  //   layer_buffer[2], // qkvWeightsBuffer,
  //   layer_buffer[3], // qkvBiasBuffer,
  //   layer_buffer[4], // linearWeightsBuffer,
  //   layer_buffer[5] // linearBiasBuffer
  // );

  const attentionResultBuffer = inlineAttention(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    attentionDotProductScale,
    layerNormAttentionOutputBuffer,
    n_heads,
    layer_buffer[2], // qkvWeightsBuffer,
    layer_buffer[3], // qkvBiasBuffer,
    layer_buffer[4], // linearWeightsBuffer,
    layer_buffer[5] // linearBiasBuffer
  );

  // const outputQKVWeights = createOutputBuffer(device, commandEncoder, layer_buffer[2], n_embd, 3 * n_embd);
  // const outputQKVBias = createOutputBuffer(device, commandEncoder, layer_buffer[3], 1, 3 * n_embd);
  // const outputQKV = createOutputBuffer(device, commandEncoder, qkvResultBuffer, seq_length, 3 * n_embd);
  // const outputQ = createOutputBuffer(device, commandEncoder, splitQResultBuffer, seq_length, n_embd);
  // const outputK = createOutputBuffer(device, commandEncoder, splitKResultBuffer, seq_length, n_embd);
  // const outputV = createOutputBuffer(device, commandEncoder, splitVResultBuffer, seq_length, n_embd);

  // const outputWeights = createOutputBuffer(device, commandEncoder, attentionWeightsResultBuffer, seq_length, seq_length * n_heads);
  // const outputAdjust = createOutputBuffer(device, commandEncoder, multiplyResultBuffer, seq_length, seq_length * n_heads);
  // const outputMask = createOutputBuffer(device, commandEncoder, causalMaskResultBuffer, seq_length, seq_length * n_heads);
  // const outputValues = createOutputBuffer(device, commandEncoder, attentionValuesResultBuffer, seq_length, n_embd);

  const outputAttentionBuffer = createOutputBuffer(device, commandEncoder, attentionResultBuffer, seq_length, n_embd);

  queue.submit([commandEncoder.finish()]);

  await outputAttentionBuffer.mapAsync(GPUMapMode.READ);
  // await outputQKV.mapAsync(GPUMapMode.READ);
  // await outputQKVWeights.mapAsync(GPUMapMode.READ);
  // await outputQKVBias.mapAsync(GPUMapMode.READ);
  // await outputQ.mapAsync(GPUMapMode.READ);
  // await outputK.mapAsync(GPUMapMode.READ);
  // await outputV.mapAsync(GPUMapMode.READ);
  // await outputWeights.mapAsync(GPUMapMode.READ);
  // await outputAdjust.mapAsync(GPUMapMode.READ);
  // await outputMask.mapAsync(GPUMapMode.READ);
  // await outputValues.mapAsync(GPUMapMode.READ);

  // const qkv_weights = new Float32Array(outputQKVWeights.getMappedRange());
  // const qkv_bias = new Float32Array(outputQKVBias.getMappedRange());
  // console.log("qkv", formatAsMatrix(new Float32Array(outputQKV.getMappedRange()), seq_length, 3 * n_embd));
  // console.log("qkv_goal", validateModel[tokenIndex][`block0_attn_catt`].data[0]);
  // console.log("qkv weights", formatAsMatrix(qkv_weights, n_embd, 3 * n_embd));
  // console.log("qkv bias", formatAsMatrix(qkv_bias, 1, 3 * n_embd));

  // const cpuTestWeights = matrixMult(layerNormAttention, formatAsMatrix(qkv_weights, n_embd, 3 * n_embd), seq_length, 3 * n_embd, n_embd);
  // console.log("cpu test weights", cpuTestWeights);
  // const cpuTestOutput = matrixAdd1dRow(cpuTestWeights, qkv_bias, seq_length, n_embd * 3);
  // console.log(cpuTestOutput);
  // console.log("q", formatAsMatrix(new Float32Array(outputQ.getMappedRange()), seq_length, n_embd));
  // console.log("k", formatAsMatrix(new Float32Array(outputK.getMappedRange()), seq_length, n_embd));
  // console.log("v", formatAsMatrix(new Float32Array(outputV.getMappedRange()), seq_length, n_embd));
  // // console.log("weights", formatAsMatrix(new Float32Array(outputWeights.getMappedRange()), seq_length, seq_length * n_heads));
  // console.log("adjust", formatAsMatrix(new Float32Array(outputAdjust.getMappedRange()), seq_length, seq_length * n_heads));
  // console.log("mask", formatAsMatrix(new Float32Array(outputMask.getMappedRange()), seq_length * n_heads, seq_length));
  // console.log("values", formatAsMatrix(new Float32Array(outputValues.getMappedRange()), seq_length, n_embd));

  // validateResult(new Float32Array(outputQKV.getMappedRange()), validateModel[tokenIndex][`block0_attn_catt`]);
  validateResult(new Float32Array(outputAttentionBuffer.getMappedRange()), validateModel[tokenIndex][`block0_attn`]);

  // throw new Error("stop");

  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);

  console.log("Result:", result);

  const resultMatrix = formatAsMatrix(new Float32Array(result), seq_length, vocab_size);
  console.log("Result matrix:");

  return resultMatrix[0];
}

function matrixMult(matA, matB, rows, cols, shared) {
  if (matA.length !== rows || matB[0].length !== cols || matA[0].length !== matB.length || matB.length !== shared) {
    console.log("matA", matA, "matB", matB, rows, cols, shared);
    throw Error("Unmatching dims for mat mul on cpu");
  }
  const output = [];
  for (let row = 0; row < rows; row++) {
    output.push([]);
    for (let col = 0; col < cols; col++) {
      let sum = 0;
      for (let i = 0; i < shared; i++) {
        sum += matA[row][i] * matB[i][col];
      }
      output[row].push(sum);
    }
  }
  return output;
}

function matrixAdd1dRow(matA, one_d, rows, cols) {
  if (matA.length !== rows || matA[0].length !== cols || one_d.length !== cols) {
    console.log("matA", matA, "one_d", one_d, rows, cols);
    throw Error("Unmatching dims for mat add 1d row on cpu");
  }
  const output = [];
  for (let row = 0; row < rows; row++) {
    output.push([]);
    for (let col = 0; col < cols; col++) {
      output[row].push(matA[row][col] + one_d[col]);
    }
  }
  return output;
}

let modelParams = null;
let tokenizer = null;

async function demo() {
  /* 
    If you're trying to run a custom model or modify the code, here's a quick guide.
    You'll need to first load a model + load a tokenizer.
      (1) Call loadModel(folder_name) with the name of a folder which contains the weights + params of your model.
      (2) Set the global tokenizer to a certain one. Pre-loaded is loadGPT2Tokenizer (most common) and loadSimpleTokenizer.
    The main entry point is generate() -> runGPT() -> various kernels.
    If you're interested in learning more about how WebGPU works, I'll likely be releasing a Youtube video/blog post on how the whole system works. Let me know if you're looking for that @ https://twitter.com/willdepue.
  */
  modelParams = await loadModel("better_shakespeare");
  tokenizer = await loadSimpleTokenizer();
  const textStream = generate("What is the answer to life, the universe, and everything?", 30);
  for await (const text of textStream) {
    console.log("Generated token:", text);
  }
}

async function* generate(prompt, max_new_tokens, top_k = 10, temperature = 1.0) {
  if (!modelParams || !tokenizer) {
    console.error("Model not loaded yet");
    return;
  }

  console.log("Starting generation with prompt", prompt);
  let history = tokenizer.encode(prompt);

  const block_size = modelParams.params.block_size;
  for (let i = 0; i < max_new_tokens; i++) {
    const idx_cond = history.slice(-block_size);

    const startTime = performance.now();
    const logits = await runGPT(idx_cond);
    const endTime = performance.now();

    console.log(`Kernel execution time: ${endTime - startTime} ms`);

    const { topKIndices, topKProbs } = selectTopK(logits, top_k);
    const probs = cpuSoftmax(topKProbs, temperature);
    const idx_next = topKIndices[sampleFromDistribution(probs)];

    history = history.concat(idx_next);

    console.log(`Output:\n${tokenizer.decode(history)}`);

    yield tokenizer.decode([idx_next]);
  }
}

async function runGPT(idx) {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeightsBuffer } = modelParams;
  const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size } = params;
  const seq_length = idx.length;

  const commandEncoder = device.createCommandEncoder();

  const embdOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  for (let i = 0; i < seq_length; i++) {
    commandEncoder.copyBufferToBuffer(
      embeddingWeightsBuffer,
      bufferSizeCalc(n_embd) * idx[i],
      embdOutputBuffer,
      bufferSizeCalc(n_embd) * i,
      bufferSizeCalc(n_embd)
    );
  }

  // Crop the position embeddings to the correct size.
  const posEmbdOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  commandEncoder.copyBufferToBuffer(
    posEmbdBuffer,
    0, // Source offset (starting from the beginning of the buffer)
    posEmbdOutputBuffer, // Destination buffer (cropped buffer)
    0, // Destination offset (starting from the beginning of the cropped buffer)
    bufferSizeCalc(seq_length, n_embd) // Number of bytes to copy
  );
  // Residual connection is just elementwise addition, can be used for combining embedding and position embedding.
  const embeddedInputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, embdOutputBuffer, posEmbdOutputBuffer);
  let layerOutputBuffer = embeddedInputBuffer;

  for (let i = 0; i < n_layer; i++) {
    const buffers = layer_buffers[i];

    const layerNormAttentionOutputBuffer = inlineLayerNorm(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      layerOutputBuffer,
      buffers.normAttentionGammaBuffer,
      buffers.normAttentionBetaBuffer
    );

    const attentionOutputBuffer = inlineAttention(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      attentionDotProductScale,
      layerNormAttentionOutputBuffer,
      n_head,
      buffers.qkvWeightsBuffer,
      buffers.qkvBiasBuffer,
      buffers.linearWeightsBuffer,
      buffers.linearBiasBuffer
    );

    const residualAttentionOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, attentionOutputBuffer, layerOutputBuffer);

    const layerNormLinearOutputBuffer = inlineLayerNorm(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      residualAttentionOutputBuffer,
      buffers.normLinearGammaBuffer,
      buffers.normLinearBetaBuffer
    );

    const linearOutputBuffer = inlineFFN(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      hidden_size,
      layerNormLinearOutputBuffer,
      buffers.firstLayerWeightsBuffer,
      buffers.firstLayerBiasBuffer,
      buffers.secondLayerWeightsBuffer,
      buffers.secondLayerBiasBuffer
    );

    const residualLinearOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

    layerOutputBuffer = residualLinearOutputBuffer;
  }

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerOutputBuffer, normGammaBuffer, normBetaBuffer);

  /* 
    Compute Pass Splitting: Possible Approaches.

    The limiting factor running larger models is often the maxStorageBufferBindingSize, which prevents you from doing giant matrix multiplications. As far as I know, the best way to solve this is to generate multiple compute passes and split the calculation into smaller sub-matrices. You can simply write back to the buffer with the proper offset and byte size and nothing changes. As long as these operations don't have inter-dependencies, WebGPU should recognize that they can be run in parallel and you shouldn't experience significant performance losses. This needs to be verified!

    The question then is to how to divide the operation properly for efficiency. I'm still figuring out how everything works, so I'm unsure what the most efficient way to do this is.

    The first thought is that if you have a matrix of elements maxStorageBufferBindingSize * 2, it's straightforward to chop it down the middle. However for non-evenly divisible matrix sizes, you might run into serious memory inefficiencies if you divide by the minimum number of sub-matrices. 

    I've implemented/planned a couple different solutions.

    (1) Start with the minimum number of groups and calculate wasted memory, then increase # of groups and record the most efficient sizing up to some maximum group number. This calculation can be done when the model is loaded.

    (2) Divide by some standard matrix size (maybe a square matrix of rows * rows) and then add one final "overflow matrix" of some irregular size. I really don't know if this is more efficient, still learning, but my gut tells me this might be result in too many groups when fewer could do better.
    
    (3) Assume that the matrix has some decently small factor that fits perfectly and use that. This is the simplest solution, and given that I have 0 clue which option is best until I test, I'm going with this for now.

  */

  const slicedEmbedOutputBuffer = createBuffer(device, bufferSizeCalc(1, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  commandEncoder.copyBufferToBuffer(
    layerNormOutputBuffer, // Source buffer (original position embeddings)
    bufferSizeCalc(seq_length - 1, n_embd), // Source offset (starting from the beginning of the buffer)
    slicedEmbedOutputBuffer, // Destination buffer (cropped buffer)
    0, // Destination offset (starting from the beginning of the cropped buffer)
    bufferSizeCalc(1, n_embd) // Number of bytes to copy
  );

  const deEmbedOutputBuffer = createBuffer(device, bufferSizeCalc(1, vocab_size), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  // Assumes that vocab_size has a decent least prime factor.
  const maxStorageBufferSize = device.limits.maxStorageBufferBindingSize;
  const totalElements = bufferSizeCalc(vocab_size, n_embd);
  var numInstances = Math.ceil(totalElements / maxStorageBufferSize);
  if (numInstances > 1) numInstances = leastPrimeFactor(vocab_size, numInstances);
  var vocabChunkSize = vocab_size / numInstances;

  for (let i = 0; i < numInstances; i++) {
    const deEmbedChunkInputBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocabChunkSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    // Remember that embeddingWeightsBuffer is transposed.
    commandEncoder.copyBufferToBuffer(
      embeddingWeightsBuffer,
      i * bufferSizeCalc(n_embd * vocabChunkSize),
      deEmbedChunkInputBuffer,
      0,
      bufferSizeCalc(n_embd, vocabChunkSize)
    );
    // We're doing some buffer tricks here. Since slicedEmbedOutputBuffer is a row matrix, we can just pretend it's a column matrix without any changes to the way it's stored. We then multiply it by the transposed embeddingWeights chunk, resulting in a column vector which, once again, we can pretend is a row vector.
    const deEmbedChunkResultBuffer = inlineMatMul(device, queue, commandEncoder, deEmbedChunkInputBuffer, slicedEmbedOutputBuffer, vocabChunkSize, 1, n_embd);
    commandEncoder.copyBufferToBuffer(deEmbedChunkResultBuffer, 0, deEmbedOutputBuffer, i * bufferSizeCalc(vocabChunkSize), bufferSizeCalc(vocabChunkSize));
  }

  queue.submit([commandEncoder.finish()]);

  await deEmbedOutputBuffer.mapAsync(GPUMapMode.READ);
  const output = deEmbedOutputBuffer.getMappedRange();

  return new Float32Array(output);
}

async function loadModel(folder) {
  if (modelParams) {
    console.error("Model already loaded");
    return;
  }
  const { device, queue } = await initializeWebGPU();
  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, 1);
  // Should be device.limits.minStorageBufferOffsetAlignment once bug is fixed.

  console.log("Loading model from folder:", folder);
  const paramsJSON = await (await fetch(`models/${folder}/params_gpt.json`)).json();
  const { block_size, n_embd, n_head, n_layer, bias, vocab_size } = paramsJSON;
  paramsJSON.hidden_size = n_embd * 4;
  paramsJSON.attentionDotProductScale = 1 / Math.sqrt(n_embd / n_head);
  const { hidden_size } = paramsJSON;
  console.log("Params:", paramsJSON);

  if (n_embd % n_head != 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

  console.log("Loading token embeddings...");
  const embeddingWeights = await loadBinaryFile("models/" + folder + "/transformer.wte.weight_gpt.bin");
  const embeddingWeightsBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(embeddingWeightsBuffer, 0, embeddingWeights);

  console.log("Loading positional embeddings...");
  const posEmbeddings = await loadBinaryFile("models/" + folder + "/transformer.wpe.weight_gpt.bin");
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(block_size, n_embd), GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

  const layer_buffers = [];
  for (let i = 0; i < n_layer; i++) {
    console.log("Loading layer", i);
    const prefix = `transformer.h.${i}.`;

    console.log("\tLoading attention layer norm...");
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.weight_gpt.bin`));
    queue.writeBuffer(normAttentionBetaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.bias_gpt.bin`));

    console.log("\tLoading qkv transform...");
    const qkv_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.weight_gpt.bin`);
    const qkv_bias = bias ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.bias_gpt.bin`) : new Array(3 * n_embd).fill(0);
    const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(qkvWeightsBuffer, 0, transposeArray(qkv_weights, 3 * n_embd, n_embd));
    queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

    console.log("\tLoading attention c_proj...");
    const linear_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.weight_gpt.bin`);
    const linear_bias = bias ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, transposeArray(linear_weights, n_embd, n_embd));
    queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

    console.log("\tLoading MLP layer norm...");
    const layerNormLinearGamma = await loadBinaryFile(`models/${folder}/${prefix}ln_2.weight_gpt.bin`);
    const layerNormLinearBeta = bias ? await loadBinaryFile(`models/${folder}/${prefix}ln_2.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
    queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);

    console.log("\tLoading MLP first layer...");
    const firstLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.weight_gpt.bin`);
    const firstLayerBias = bias ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.bias_gpt.bin`) : new Array(hidden_size).fill(0);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, transposeArray(firstLayerWeights, hidden_size, n_embd));
    queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);

    console.log("\tLoading MLP second layer...");
    const secondLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.weight_gpt.bin`);
    const secondLayerBias = bias ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, transposeArray(secondLayerWeights, n_embd, hidden_size));
    queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);

    layer_buffers.push({
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
    });
  }

  console.log("Loading final layer norm...");
  const layerNormGamma = await loadBinaryFile(`models/${folder}/transformer.ln_f.weight_gpt.bin`);
  const layerNormBeta = bias ? await loadBinaryFile(`models/${folder}/transformer.ln_f.bias_gpt.bin`) : new Array(n_embd).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
  queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

  const output = {
    device,
    queue,
    params: paramsJSON,
    layer_buffers,
    embeddingWeightsBuffer,
    posEmbdBuffer,
    normGammaBuffer,
    normBetaBuffer,
  };

  console.log("Finished loading model.", output);
  return output;
}

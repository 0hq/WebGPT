function transformerBlock(device, queue, commandEncoder, seq_length, n_embd, n_heads, inputBuffer) {
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.

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

  const layerNormAttentionOutputBuffer = inlineLayerNorm(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    inputBuffer,
    normAttentionGammaBuffer,
    normAttentionBetaBuffer
  );

  const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  queue.writeBuffer(qkvWeightsBuffer, 0, qkv_weights);
  queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

  const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  queue.writeBuffer(linearWeightsBuffer, 0, linear_weights);
  queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

  const attentionOutputBuffer = inlineAttention(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    layerNormAttentionOutputBuffer,
    n_heads,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer
  );

  const residualAttentionOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, attentionOutputBuffer, inputBuffer);

  const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
  queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);

  const layerNormLinearOutputBuffer = inlineLayerNorm(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    residualAttentionOutputBuffer,
    normLinearGammaBuffer,
    normLinearBetaBuffer
  );

  const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(firstLayerWeightsBuffer, 0, firstLayerWeights);
  queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);

  const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(secondLayerWeightsBuffer, 0, secondLayerWeights);
  queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);

  const linearOutputBuffer = inlineFFN(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    hidden_size,
    layerNormLinearOutputBuffer,
    firstLayerWeightsBuffer,
    firstLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerBiasBuffer
  );

  const residualLinearOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

  return residualLinearOutputBuffer;
}

// (async () => {
//   const { device, queue } = await initializeWebGPU();
//   const commandEncoder = device.createCommandEncoder();
//   const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
//   const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

//   const seq_length = 1024;
//   const n_embd = 768;
//   const n_heads = 4;
//   const inputMatrix = new Float32Array(seq_length * n_embd);
//   for (let i = 0; i < inputMatrix.length; i++) {
//     inputMatrix[i] = i;
//   }
//   const inputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
//   queue.writeBuffer(inputBuffer, 0, inputMatrix);

//   const blockOutputBuffer = transformerBlock(device, queue, commandEncoder, seq_length, n_embd, n_heads, inputBuffer);

//   const output_rows = seq_length;
//   const output_cols = n_embd;
//   const outputBufferSize = bufferSizeCalc(output_rows, output_cols);

//   const outputBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
//   const copyCommandEncoder = device.createCommandEncoder();
//   copyCommandEncoder.copyBufferToBuffer(blockOutputBuffer, 0, outputBuffer, 0, outputBufferSize);

//   queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

//   await outputBuffer.mapAsync(GPUMapMode.READ);
//   const result = outputBuffer.getMappedRange();
//   printMatrix(output_rows, output_cols, new Float32Array(result));
// })();

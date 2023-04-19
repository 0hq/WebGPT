function transformerBlock(
  device,
  queue,
  commandEncoder,
  seq_length,
  n_embd,
  n_heads,
  attentionDotProductScale,
  inputBuffer,
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
  secondLayerBiasBuffer
) {
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.

  console.log("\tNormalizing block input...");

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

  console.log("\tRunning attention block...");

  const attentionOutputBuffer = inlineAttention(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    attentionDotProductScale,
    layerNormAttentionOutputBuffer,
    n_heads,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer
  );

  console.log("\tResidual connections from attention and input...");

  const residualAttentionOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, attentionOutputBuffer, inputBuffer);

  console.log("\tNormalizing attention output...");

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

  console.log("\tRunning MLP...");

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

  console.log("\tResidual connections from MLP and attention...");

  const residualLinearOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

  return {
    layerNormAttentionOutputBuffer,
    attentionOutputBuffer,
    residualAttentionOutputBuffer,
    layerNormLinearOutputBuffer,
    linearOutputBuffer,
    residualLinearOutputBuffer,
  };
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

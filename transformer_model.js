async function runGPT() {
  const { device, queue } = await initializeWebGPU();
  const limits = device.limits;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const commandEncoder = device.createCommandEncoder();
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const context_size = 1024;
  const seq_length = 24;
  const vocab_size = 50304;
  const n_embd = 768 / 2;
  const n_heads = 4;
  const n_layers = 12;
  const inputMatrix = new Float32Array(seq_length * vocab_size).fill(1);

  let layerBuffer = null;

  layerBuffer = createBuffer(device, bufferSizeCalc(seq_length, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(layerBuffer, 0, inputMatrix);

  const embeddings = new Float32Array(vocab_size * n_embd).fill(-1);
  const embdBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdBuffer, 0, embeddings);

  const posEmbeddings = new Float32Array(context_size * n_embd).fill(-1);
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

  const chunkSize = Math.floor(maxBindingSize / (n_embd * 4)); // Calculate the chunk size based on the maximum binding size
  const numChunks = Math.ceil(seq_length / chunkSize); // Determine the number of chunks required

  const chunkOutputBuffers = [];
  for (let i = 0; i < numChunks; i++) {
    const outputBuffer = createBuffer(device, bufferSizeCalc(chunkSize, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    chunkOutputBuffers.push(outputBuffer);
  }

  const embdOutputBuffer = inlineMatMul(device, queue, commandEncoder, layerBuffer, embdBuffer, seq_length, n_embd, vocab_size);

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
  layerBuffer = embeddedInputBuffer;

  for (let i = 0; i < n_layers; i++) {
    const blockOutputBuffer = transformerBlock(device, queue, commandEncoder, seq_length, n_embd, n_heads, layerBuffer);
    layerBuffer = blockOutputBuffer;
  }

  const layerNormGamma = new Float32Array(seq_length).fill(1);
  const layerNormBeta = new Float32Array(seq_length).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(seq_length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(seq_length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
  queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerBuffer, normGammaBuffer, normBetaBuffer);

  const deEmbeddings = new Float32Array(n_embd * vocab_size).fill(-1);
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, deEmbeddings);

  const deEmbedOutputBuffer = inlineMatMul(device, queue, commandEncoder, layerNormOutputBuffer, deEmbedBuffer, seq_length, vocab_size, n_embd);

  const output_rows = seq_length;
  const output_cols = vocab_size;
  const outputBufferSize = bufferSizeCalc(output_rows, output_cols);
  const outputBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(deEmbedOutputBuffer, 0, outputBuffer, 0, outputBufferSize);
  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);
  await outputBuffer.mapAsync(GPUMapMode.READ);
  const result = outputBuffer.getMappedRange();
  printMatrix(output_rows, output_cols, new Float32Array(result));
}

async function timeGPT() {
  const startTime = performance.now();
  await runGPT();
  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);
}

// timeGPT();

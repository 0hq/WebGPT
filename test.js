class TestShader {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;

    this.unloadDeletionStack = [];
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    initializeOperations(this.device);

    this.initialized = true;

    console.log("Model initialized");
  }

  async test() {
    // ---------------- Create Passes ---------------- //
    const { M, N } = { M: 100, N: 100 };
    const input_array = new Float32Array(M * N).fill(1);
    const weight_array = new Float32Array(M * N).fill(1);

    const inputBuffer = this.initTensor(input_array, [M, N], ["storage"]);
    const weightBuffer = this.initTensor(weight_array, [M, N], ["storage"]);

    this.computePasses = [];
    let intermediateBuffer = inputBuffer;
    {
      const { passes, resultBuffer } = ResidualBlock.newInstance(M, N, intermediateBuffer, weightBuffer);
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    {
      const { passes, resultBuffer } = OutputBlock.newInstance(M, N, intermediateBuffer);
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    const resultBuffer = intermediateBuffer;

    // ---------------- Compute Passes ----------------

    const commandEncoder = this.device.createCommandEncoder();
    for (const pass of this.computePasses) {
      if (pass.flag === "compute") {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pass.pipeline);
        for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
        passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y);
        passEncoder.end();
      } else if (pass.flag === "copy") {
        commandEncoder.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
      }
    }
    this.device.queue.submit([commandEncoder.finish()]);

    // ---------------- Read Results ----------------

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const output = resultBuffer.getMappedRange();
    const outputArray = new Float32Array(output).slice(0); // Copy the array, otherwise it'll be destroyed.
    console.log(formatAsMatrix(outputArray, M, N));

    destroyOperationBuffers();
    this.unloadBuffers();

    return outputArray;
  }

  initTensor(data, dims, ops) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    return Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
  }
}

async function testInstruction() {
  const testShader = new TestShader();
  await testShader.initialize();
  await testShader.test();
}

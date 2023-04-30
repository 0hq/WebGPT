class Instruction {
  constructor(device) {
    this.device = device;
    this.bufferDeletionStack = [];
    this.unloadDeletionStack = [];

    this.initBindGroups();
  }

  initBindGroup(layout, buffers, label = "") {
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
      label,
    });
  }

  initBuffer(ops, row, col = 1, noDelete = false) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(row, col),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    if (!noDelete) this.bufferDeletionStack.push(buffer);
    else this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  bufferSize(dimA, dimB = 1) {
    return Math.ceil((dimA * dimB * Float32Array.BYTES_PER_ELEMENT) / 1) * 1;
  }

  initBindGroups() {
    const bg = (types) =>
      this.device.createBindGroupLayout({
        entries: types.map((entry, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: entry },
        })),
      });

    this.r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_Layout = bg(["read-only-storage", "read-only-storage"]);
    this.r_Layout = bg(["read-only-storage"]);
    this.u_s_Layout = bg(["uniform", "storage"]);
    this.u_s_s_s_Layout = bg(["uniform", "storage", "storage", "storage"]);
  }

  initPipeline(code, bindGroupLayouts, label = "", constants = {}) {
    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts }),
      compute: {
        module: this.device.createShaderModule({ code }),
        entryPoint: "main",
        constants,
      },
      label,
    });
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  destroyBuffers() {
    this.bufferDeletionStack.map((buffer) => buffer.destroy());
    this.bufferDeletionStack = [];
  }
}

class FastMatMul extends Instruction {
  constructor(device) {
    super(device);
    this.name = "fastMatMul";
    this.pipelineCache = new Map();
  }

  getPipeline(rows) {
    const div4 = rows % 4 === 0;
    const pipelineCacheKey = div4 ? "fastMatMulNoCheck" : "fastMatMul";
    if (this.pipelineCache.has(pipelineCacheKey)) {
      return this.pipelineCache.get(pipelineCacheKey);
    }
    const kernel = div4 ? this.fastMatMulNoCheck : this.fastMatMul;
    const pipeline = this.initPipeline(kernel, [this.u_s_Layout, this.r_r_Layout], pipelineCacheKey);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, bufA, bufB) {
    const pipeline = this.getPipeline(rows);
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const resultBuf = this.initBuffer(["storage", "copy_from"], rows, cols);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuf], "opBindGroup");
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [bufA, bufB], "inputBindGroup");
    const workgroups = { x: wgSize(cols, 64), y: wgSize(rows, 32) };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, Math.ceil(cols / 4), Math.ceil(shared / 4)]));

    return {
      resultBuf,
      pass: {
        pipeline,
        groups: [opBindGroup, inputBindGroup],
        workgroups,
      },
    };
  }
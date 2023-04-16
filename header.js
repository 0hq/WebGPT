async function initializeWebGPU() {
  if (!navigator.gpu) {
    console.error("WebGPU is not supported");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const queue = device.queue;

  return { device, queue };
}

function createShader(device, code) {
  return device.createShaderModule({
    code,
  });
}

function createBindGroupLayout(device, string_entries) {
  const entries = string_entries.map((entry, i) => ({
    binding: i,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: entry },
  }));
  return device.createBindGroupLayout({
    entries,
  });
}

function createPipelineLayout(device, bindGroupLayouts) {
  return device.createPipelineLayout({
    bindGroupLayouts,
  });
}

function createComputePipeline(device, shaderModule, pipelineLayout) {
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });
}

function createPipeline(device, shaderString, bindGroupLayouts) {
  const shaderModule = createShader(device, shaderString);
  const pipelineLayout = createPipelineLayout(device, bindGroupLayouts);
  const pipeline = createComputePipeline(device, shaderModule, pipelineLayout);
  return pipeline;
}

function createBindGroup(device, bindGroupLayout, buffers) {
  const entries = buffers.map((buffer, i) => ({
    binding: i,
    resource: {
      buffer,
    },
  }));
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries,
  });
}

function createBuffer(device, size, usage) {
  return device.createBuffer({
    size: size,
    usage: usage,
  });
}

function createOutputBuffer(device, commandEncoder, buffer, rows, cols) {
  const outputBufferSize = bufferSizeCalc(rows, cols);
  const outputBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  commandEncoder.copyBufferToBuffer(buffer, 0, outputBuffer, 0, outputBufferSize);
  return outputBuffer;
}

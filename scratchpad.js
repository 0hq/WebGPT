const mainShader = `
struct CMeta {
  M: u32,
  N: u32,
  K: u32,
  MD4: u32,
  ND4: u32,
  KD4: u32,
  origM: u32,
  origN: u32,
  origK: u32,
}

@group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
@group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

@group(0) @binding(0) var<uniform> cmeta: CMeta;
@group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

fn selectValueFromVec4(v: vec4<f32>, idx: u32) -> f32 {
  var result: f32;
  if (idx == 0u) {
    result = v.x;
  } else if (idx == 1u) {
    result = v.y;
  } else if (idx == 2u) {
    result = v.z;
  } else {
    result = v.w;
  }
  return result;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  var ND4: u32 = cmeta.ND4;
  var KD4: u32 = cmeta.KD4;
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;

  if (x >= cmeta.origN || y >= cmeta.origM) {
    return;
  }

  var sum00: vec4<f32> = vec4<f32>();
  var sum01: vec4<f32> = vec4<f32>();
  var sum02: vec4<f32> = vec4<f32>();
  var sum03: vec4<f32> = vec4<f32>();
  var sum10: vec4<f32> = vec4<f32>();
  var sum11: vec4<f32> = vec4<f32>();
  var sum12: vec4<f32> = vec4<f32>();
  var sum13: vec4<f32> = vec4<f32>();
  for(var k: u32 = 0u; k < KD4; k = k + 1u) {
    var arow0: vec4<f32> = vec4<f32>();
    var arow1: vec4<f32> = vec4<f32>();
    var arow2: vec4<f32> = vec4<f32>();
    var arow3: vec4<f32> = vec4<f32>();
    
    arow0 = array_a[(y * 4u + 0u) * KD4 + k];
    arow1 = array_a[(y * 4u + 1u) * KD4 + k];
    arow2 = array_a[(y * 4u + 2u) * KD4 + k];
    arow3 = array_a[(y * 4u + 3u) * KD4 + k];

    for (var ki: u32 = 0u; ki < 4u; ki = ki + 1u) {
      var brow: vec4<f32>;
  
      let arow0_val = selectValueFromVec4(arow0, ki);
      let arow1_val = selectValueFromVec4(arow1, ki);
      let arow2_val = selectValueFromVec4(arow2, ki);
      let arow3_val = selectValueFromVec4(arow3, ki);
  
      brow = array_b[(k * 4u + ki) * ND4 + x * 2u + 0u];
      sum00 = vec4<f32>(arow0_val) * brow + sum00;
      sum01 = vec4<f32>(arow1_val) * brow + sum01;
      sum02 = vec4<f32>(arow2_val) * brow + sum02;
      sum03 = vec4<f32>(arow3_val) * brow + sum03;
  
      brow = array_b[(k * 4u + ki) * ND4 + x * 2u + 1u];
      sum10 = vec4<f32>(arow0_val) * brow + sum10;
      sum11 = vec4<f32>(arow1_val) * brow + sum11;
      sum12 = vec4<f32>(arow2_val) * brow + sum12;
      sum13 = vec4<f32>(arow3_val) * brow + sum13;
    }
  }

  array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
  array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
  array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
  array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
  array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
  array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
  array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
  array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
  
}`;

class TestGPU {
  constructor() {
    this.initialized = false;
    this.device = null;
    this.minStorageBufferOffsetAlignment = 1;
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    this.initBindGroups();
    this.initPipelines();

    this.initialized = true;
  }

  padMatrixToVec4(matrix, rows, cols) {
    const rowsPadded = Math.ceil(rows / 4) * 4;
    const colsPadded = Math.ceil(cols / 4) * 4;
    const paddedMatrix = new Float32Array(rowsPadded * colsPadded);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        paddedMatrix[i * colsPadded + j] = matrix[i * cols + j];
      }
    }

    return [paddedMatrix, this.bufferSizeCalc(rowsPadded, colsPadded)];
  }

  async testMatmul() {
    const commandEncoder = this.device.createCommandEncoder();

    const dim = 400;
    const identityMatrix = new Float32Array(dim * dim).fill(1);
    // for (let i = 0; i < dim; i++) {
    //   identityMatrix[i * dim + i] = 1;
    // }
    const [paddedIdentityMatrix, bufferSizeA] = this.padMatrixToVec4(identityMatrix, dim, dim);

    const matrixABuffer = createBuffer(this.device, dim * dim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(matrixABuffer, 0, identityMatrix);

    const matrixB = new Float32Array(dim * dim).fill(1);
    const [paddedMatrixB, bufferSizeB] = this.padMatrixToVec4(matrixB, dim, dim);

    const matrixBBuffer = createBuffer(this.device, dim * dim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(matrixBBuffer, 0, matrixB);

    // const matmulResultBuffer = this.inlineFastMatMul(commandEncoder, matrixABuffer, matrixBBuffer, dim, dim, dim);
    // const matmulResultBuffer = this.inlineAltMatMul(commandEncoder, matrixABuffer, matrixBBuffer, dim, dim, dim);
    const matmulResultBuffer = this.inlineMainMatMul(commandEncoder, matrixABuffer, matrixBBuffer, dim, dim, dim);
    // const matmulResultBuffer = this.inlineMatMul(commandEncoder, matrixABuffer, matrixBBuffer, dim, dim, dim);

    const outputBuffer = createOutputBuffer(this.device, commandEncoder, matmulResultBuffer, dim, dim);

    this.device.queue.submit([commandEncoder.finish()]);

    await outputBuffer.mapAsync(GPUMapMode.READ);
    const output = outputBuffer.getMappedRange();

    console.log(formatAsMatrix(new Float32Array(output), dim, dim));
    console.log(new Float32Array(output));
  }

  inlineMainMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    console.log("inlineFastMatMul", rows, cols, shared);

    const rowsPadded = Math.ceil(rows / 4) * 4;
    const colsPadded = Math.ceil(cols / 4) * 4;
    const sharedPadded = Math.ceil(shared / 4) * 4;

    const matmulUniformBuffer = createBuffer(this.device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const matmulResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const matMulBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(
      matmulUniformBuffer,
      0,
      new Uint32Array([rowsPadded, colsPadded, sharedPadded, Math.ceil(rows / 4), Math.ceil(cols / 4), Math.ceil(shared / 4), rows, cols, shared])
    );

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.fastMatMulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, 8), workgroupCalc(cols, 8));
    passEncoder.end();

    return matmulResultBuffer;
  }

  inlineMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    const matmulUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const matmulResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const matMulBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, shared]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.matmulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder.end();

    return matmulResultBuffer;
  }

  /*
    const request: WebGPURunnerRequest = {
      pipeline,
      buffers: [
        { index: 0, name: 'array_a', length: m * k, input: true, output: false },
        { index: 1, name: 'array_b', length: k * n, input: true, output: false },
        { index: 2, name: 'array_c', length: m * n, input: false, output: true },
        { index: 3, name: 'meta', length: 7, input: true, output: false },
      ],
      inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, m / 4, n / 4, k / 4, alpha]) },
      threadGroups: { x: n / 64, y: m / 32, z: 1 }
    };
    */

  inlineAltMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    const matmulUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const matmulResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const matMulBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, shared]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.altFastMatMulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder.end();

    return matmulResultBuffer;
  }

  inlineFastMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    console.log("inlineFastMatMul", rows, cols, shared);

    const rowsPadded = Math.ceil(rows / 4) * 4;
    const colsPadded = Math.ceil(cols / 4) * 4;
    const sharedPadded = Math.ceil(shared / 4) * 4;

    const matmulUniformBuffer = createBuffer(this.device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const matmulResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const matMulBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(
      matmulUniformBuffer,
      0,
      new Uint32Array([rowsPadded, colsPadded, sharedPadded, Math.ceil(rows / 4), Math.ceil(cols / 4), Math.ceil(shared / 4), rows, cols, shared])
    );

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.fastMatMulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, 8), workgroupCalc(cols, 8));
    passEncoder.end();

    return matmulResultBuffer;
  }

  bufferSizeCalc(dimA, dimB = 1) {
    return alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, this.minStorageBufferOffsetAlignment);
  }

  initBindGroups() {
    this.r_r_BindLayout = createBindGroupLayout(this.device, ["read-only-storage", "read-only-storage"]);
    this.u_s_BindLayout = createBindGroupLayout(this.device, ["uniform", "storage"]);
  }

  initPipelines() {
    this.matmulPipeline = createPipeline(this.device, matMulShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
    this.fastMatMulPipeline = createPipeline(this.device, fastMatMulShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
    this.altFastMatMulPipeline = createPipeline(this.device, altFastMatMulShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
    this.mainPipeline = createPipeline(this.device, mainShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
  }
}

(async () => {
  const GPU = new TestGPU();
  await GPU.initialize();
  await GPU.testMatmul();
})();

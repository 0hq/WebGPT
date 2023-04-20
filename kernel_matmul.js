function createMatMulShader() {
  return `
    struct Matrix {
        data: array<f32>, 
    }

    struct Uniforms {
      dimY: u32, // row dimension of A and row dimension of C
      dimX: u32, // col dimension of B and col dimension of C
      dimS: u32, // shared dimension of A and B
    };

    @group(1) @binding(0) var<storage, read> A: Matrix;
    @group(1) @binding(1) var<storage, read> B: Matrix;

    @group(0) @binding(1) var<storage, read_write> C: Matrix;
    @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row: u32 = global_id.x;
        let col: u32 = global_id.y;
        let dimX: u32 = dimBuffer.dimX;
        let dimY: u32 = dimBuffer.dimY;
        let dimS: u32 = dimBuffer.dimS;

        if (row >= dimY || col >= dimX) {
          return;
        }

        var sum: f32 = 0.0;
        for (var i: u32 = 0; i < dimS; i = i + 1) {
            sum = sum + A.data[row * dimS + i] * B.data[i * dimX + col];
        }

        C.data[row * dimX + col] = sum;
      } 
  `;
}

function inlineMatMul(device, queue, commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, can be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage", "read-only-storage"]);

  const matmulBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const matmulPipeline = createPipeline(device, createMatMulShader(), [matmulBindGroupLayout, inputBufferBindGroupLayout]);

  const matmulUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const matmulResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const matMulBindGroup = createBindGroup(device, matmulBindGroupLayout, [matmulUniformBuffer, matmulResultBuffer]);
  queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, shared]));

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(matmulPipeline);
  passEncoder.setBindGroup(0, matMulBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [Abuffer, Bbuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder.end();

  return matmulResultBuffer;
}

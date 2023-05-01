/**
 * This is a sample script with an int8 gemm shader. I don't have time to add it to the repo's system, so I'm just going to leave it here.
 * The current fastmatmul kernel gets 1.2 GFLOPs on an M1 Pro for the M, N, and K below, whereas my kernel gets 4.2 GFLOPs. This should greatly speed up inference and also speed up model loading time.
 */


const M = 1;
const N = 2048;
const K = 2048;
const workgroupSizeX = 16;
const workgroupSizeY = 16;

const A = new Float32Array(M * K);
const B = new Float32Array(K * N);
const C = new Float32Array(M * N);

// Initialize matrices A and B with random values
for (let i = 0; i < M * K; i++) {
    A[i] = ((Math.random() * 2) - 1) / 5;
}
for (let i = 0; i < K * N; i++) {
    B[i] = ((Math.random() * 2) - 1) / 5;
}


function quantizeMatrix(matrix, M, N) {
    const blockSize = 4;
    const quantizedMatrix = new Int32Array(Math.ceil(M * N / blockSize));

    // Find the global absmax value
    let absmax = 0;
    for (let i = 0; i < M * N; i++) {
        absmax = Math.max(absmax, Math.abs(matrix[i]));
    }

    absmax = 2.0;

    // Quantize the matrix values to int8 and pack them into Int32Array
    for (let i = 0; i < M * N; i += blockSize) {
        const packedValue =
            (Math.round(matrix[i] / absmax * 127) & 0xFF) |
            ((Math.round(matrix[i + 1] / absmax * 127) & 0xFF) << 8) |
            ((Math.round(matrix[i + 2] / absmax * 127) & 0xFF) << 16) |
            ((Math.round(matrix[i + 3] / absmax * 127) & 0xFF) << 24);
        quantizedMatrix[Math.floor(i / blockSize)] = packedValue;
    }

    return { quantizedMatrix, absmax };
}


function dequantizeMatrix(quantizedMatrix, absmax, M, N) {
    const blockSize = 4;
    const matrix = new Float32Array(M * N);

    // Dequantize the matrix values from Int32Array to Float32Array
    for (let i = 0; i < M * N; i += blockSize) {
        const packedValue = quantizedMatrix[Math.floor(i / blockSize)];
        matrix[i] = ((packedValue << 24) >> 24) / 127.0 * absmax;
        matrix[i + 1] = ((packedValue << 16) >> 24) / 127.0 * absmax;
        matrix[i + 2] = ((packedValue << 8) >> 24) / 127.0 * absmax;
        matrix[i + 3] = (packedValue >> 24) / 127.0 * absmax;
    }

    return matrix;
}

const qa = quantizeMatrix(A, M, K);
const qb = quantizeMatrix(B, K, N);

const quantizedA = qa.quantizedMatrix;
const quantizedB = qb.quantizedMatrix;

const dqB = dequantizeMatrix(quantizedB, qb.absmax, K, N);


// for (let i = 0; i < 10; i++) {
//     console.log(B[i], dqB[i]);
// }

const absmax = Math.max(qa.absmax, qb.absmax);

// Naive CPU implementation of matrix multiplication
function multiplyMatrices(A, B, C, M, N, K) {
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

async function run() {
    // Create WebGPU device and queue
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const queue = device.queue;

    // Create buffers for matrices A, B, and C
    const aBuffer = device.createBuffer({
        size: A.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bBuffer = device.createBuffer({
        size: quantizedB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const cBuffer = device.createBuffer({
        size: C.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Copy matrices A and B to their respective buffers
    queue.writeBuffer(aBuffer, 0, A);
    queue.writeBuffer(bBuffer, 0, quantizedB);

    // Create bind group layout and bind group


    const shaderCode = `
    
    @group(0) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(0) @binding(1) var<storage,read> array_b: array<i32>;

    @group(0) @binding(2) var<storage,read_write> array_c: array<vec4<f32>>;

    const absmax = ${absmax};

    fn unpackInt8x4(value: i32) -> vec4<f32> {
        let x = f32((value << 24) >> 24) / 127.0 * absmax;
        let y = f32(((value << 16) >> 24)) / 127.0 * absmax;
        let z = f32(((value << 8) >> 24)) / 127.0 * absmax;
        let w = f32(((value >> 24))) / 127.0 * absmax;
        return vec4<f32>(x, y, z, w);
    }

    @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY})
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        var M: u32 = ${M};
        var N: u32 = ${N};
        var ND4: u32 = ${Math.ceil(N / 4)};
        var KD4: u32 = ${Math.ceil(K / 4)};
        var x: u32 = global_id.x;
        var y: u32 = global_id.y;

        if (x * 8 >= N || y * 4 >= M) {
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
            var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
            var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
            var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
            var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
            var brow: vec4<f32>;

            brow = unpackInt8x4(array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u]);
            sum00 = vec4<f32>(arow0.x) * brow + sum00;
            sum01 = vec4<f32>(arow1.x) * brow + sum01;
            sum02 = vec4<f32>(arow2.x) * brow + sum02;
            sum03 = vec4<f32>(arow3.x) * brow + sum03;

            brow = unpackInt8x4(array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u]);
            sum10 = vec4<f32>(arow0.x) * brow + sum10;
            sum11 = vec4<f32>(arow1.x) * brow + sum11;
            sum12 = vec4<f32>(arow2.x) * brow + sum12;
            sum13 = vec4<f32>(arow3.x) * brow + sum13;

            brow = unpackInt8x4(array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u]);
            sum00 = vec4<f32>(arow0.y) * brow + sum00;
            sum01 = vec4<f32>(arow1.y) * brow + sum01;
            sum02 = vec4<f32>(arow2.y) * brow + sum02;
            sum03 = vec4<f32>(arow3.y) * brow + sum03;

            brow = unpackInt8x4(array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u]);
            sum10 = vec4<f32>(arow0.y) * brow + sum10;
            sum11 = vec4<f32>(arow1.y) * brow + sum11;
            sum12 = vec4<f32>(arow2.y) * brow + sum12;
            sum13 = vec4<f32>(arow3.y) * brow + sum13;

            brow = unpackInt8x4(array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u]);
            sum00 = vec4<f32>(arow0.z) * brow + sum00;
            sum01 = vec4<f32>(arow1.z) * brow + sum01;
            sum02 = vec4<f32>(arow2.z) * brow + sum02;
            sum03 = vec4<f32>(arow3.z) * brow + sum03;

            brow = unpackInt8x4(array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u]);
            sum10 = vec4<f32>(arow0.z) * brow + sum10;
            sum11 = vec4<f32>(arow1.z) * brow + sum11;
            sum12 = vec4<f32>(arow2.z) * brow + sum12;
            sum13 = vec4<f32>(arow3.z) * brow + sum13;

            brow = unpackInt8x4(array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u]);
            sum00 = vec4<f32>(arow0.w) * brow + sum00;
            sum01 = vec4<f32>(arow1.w) * brow + sum01;
            sum02 = vec4<f32>(arow2.w) * brow + sum02;
            sum03 = vec4<f32>(arow3.w) * brow + sum03;

            brow = unpackInt8x4(array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u]);
            sum10 = vec4<f32>(arow0.w) * brow + sum10;
            sum11 = vec4<f32>(arow1.w) * brow + sum11;
            sum12 = vec4<f32>(arow2.w) * brow + sum12;
            sum13 = vec4<f32>(arow3.w) * brow + sum13;
        }

        if (y * 4u + 0u < M) {
            array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
            array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
        }
        if (y * 4u + 1u < M) {
            array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
            array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
        }
        if (y * 4u + 2u < M) {
            array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
            array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
        }
        if (y * 4u + 3u < M) {
            array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
            array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
        }
    }
`;





    const shaderModule = device.createShaderModule({
        code: shaderCode,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage',
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage',
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage',
                },
            },
        ],
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: aBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: bBuffer,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: cBuffer,
                },
            },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });
    const encoder = device.createCommandEncoder();
    const passEncoder = encoder.beginComputePass();

    // Dispatch the compute kernel
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupSizeX, workgroupSizeY, 1);
    passEncoder.end()

    const readBuffer = device.createBuffer({
        size: C.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Copy matrix C from the GPU to the CPU
    encoder.copyBufferToBuffer(cBuffer, 0, readBuffer, 0, C.byteLength);

    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const readBufferData = new Float32Array(readBuffer.getMappedRange());

    const C_cpu = new Float32Array(M * N)
    multiplyMatrices(A, B, C_cpu, M, N, K);

    for (let i = 0; i < M * N; i++) {
        if (Math.abs(C_cpu[i] - readBufferData[i]) > 0.1) {
            console.error("CPU and GPU results differ at index", i);
            console.error("CPU:", C_cpu[i], "GPU:", readBufferData[i]);
            break;
        }
        // } else {
        //     console.log("CPU and GPU results are the same at index", i);
        //     console.log("CPU:", C_cpu[i], "GPU:", readBufferData[i]);
        // }
    }

    let mae = 0;
    for (let i = 0; i < M * N; i++) {
        mae += Math.abs(C_cpu[i] - readBufferData[i]);
    }
    mae /= M * N;
    console.log("Mean Absolute Error:", mae);

    const NUM_RUNS = 100;

    //warmup

    for (let i = 0; i < NUM_RUNS; i++) {

        // Dispatch the compute kernel
        const encoder = device.createCommandEncoder();
        const passEncoder = encoder.beginComputePass();

        // Dispatch the compute kernel
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(workgroupSizeX, workgroupSizeY, 1);

        passEncoder.end()

        const readBuffer = device.createBuffer({
            size: C.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Copy matrix C from the GPU to the CPU
        encoder.copyBufferToBuffer(cBuffer, 0, readBuffer, 0, C.byteLength);


    }

    // Run GPU kernel NUM_RUNS times and measure time
    let totalTime = 0;
    for (let i = 0; i < NUM_RUNS; i++) {
        const start = performance.now();

        // Dispatch the compute kernel
        const encoder = device.createCommandEncoder();
        const passEncoder = encoder.beginComputePass();

        // Dispatch the compute kernel
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(M / workgroupSizeX, N / workgroupSizeY, 1);

        passEncoder.end()

        const readBuffer = device.createBuffer({
            size: C.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Copy matrix C from the GPU to the CPU
        encoder.copyBufferToBuffer(cBuffer, 0, readBuffer, 0, C.byteLength);


        const end = performance.now();
        totalTime += end - start;
    }
    const averageTime = totalTime / NUM_RUNS;
    console.log(`Average time per run: ${averageTime.toFixed(2)} ms`);
    // print flops

    const flops = 2 * M * N * K / (averageTime);
    console.log(`GFLOPS: ${flops / 1e9}`);
}

run();
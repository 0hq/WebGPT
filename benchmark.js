async function benchmark(device, pipeline, N, workgroupSize, it = 1) {
  const A = new Array(N).fill(null).map(() => new Array(N).fill(null).map(() => Math.random()));
  const B = new Array(N).fill(null).map(() => new Array(N).fill(null).map(() => Math.random()));

  const startTime = performance.now();

  for (let i = 0; i < it; i++) {
    await runMatMul(device, pipeline, A, B);
  }

  const endTime = performance.now();

  const elapsedTime = endTime - startTime;
  const gflops = (2 * N * N * N * it) / (elapsedTime * 1e6);

  console.log(`N: ${N}, Workgroup Size: ${workgroupSize}x${workgroupSize}, Time: ${elapsedTime.toFixed(2)}ms, GFLOPS: ${gflops.toFixed(2)}`);
}

function generateRandomMatrix(rows, cols) {
  const matrix = new Array(rows);
  for (let i = 0; i < rows; i++) {
    matrix[i] = new Array(cols);
    for (let j = 0; j < cols; j++) {
      matrix[i][j] = Math.random() * 2 - 1;
    }
  }
  return matrix;
}

function matMulCPU(A, B) {
  const result = new Array(A.length);
  for (let i = 0; i < A.length; i++) {
    result[i] = new Array(B[0].length).fill(0);
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < A[0].length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

async function benchmarkMatrixMultiplication(device, pipeline) {
  const numTests = 100;
  const maxDimension = 1000;
  const rowsA = 1;
  const colsA = maxDimension;
  const rowsB = colsA;
  const colsB = 1;

  const A = generateRandomMatrix(rowsA, colsA);
  const B = generateRandomMatrix(rowsB, colsB);

  console.log("Running matrix multiplication benchmark...");
  console.log(`Matrix A: ${rowsA}x${colsA}`);
  console.log(`Matrix B: ${rowsB}x${colsB}`);

  const start = performance.now();
  for (let t = 0; t < numTests; t++) {
    const gpuResult = await runMatMul(device, pipeline, A, B);

    // console.log(`Run ${t + 1}: DONE`);
  }
  console.log("DONE");

  const end = performance.now();
  console.log(`Time taken: ${end - start} ms`);
  console.log(`Time per run: ${(end - start) / numTests} ms`);
  console.log(`Runs per second: ${numTests / ((end - start) / 1000)}`);
  console.log(`Ops per second: ${(maxDimension ** 2 * numTests) / ((end - start) / 1000)}`);
  console.log(`GFLOPS: ${(maxDimension ** 2 * numTests) / ((end - start) / 1000) / 1e9}`);
}

async function testMatrixMultiplication(device, pipeline) {
  const numTests = 100;
  const maxDimension = 2000;

  for (let t = 0; t < numTests; t++) {
    const rowsA = 1;
    const colsA = Math.ceil(Math.random() * maxDimension);
    const rowsB = colsA;
    const colsB = 1;

    const A = generateRandomMatrix(rowsA, colsA);
    const B = generateRandomMatrix(rowsB, colsB);

    const gpuResult = await runMatMul(device, pipeline, A, B);
    const cpuResult = matMulCPU(A, B);

    const epsilon = 1e-3;
    let success = true;
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        if (Math.abs(cpuResult[i][j] - gpuResult[i * colsB + j]) > epsilon) {
          success = false;
          break;
        }
      }
      if (!success) break;
    }

    console.log(`Test ${t + 1}: ${success ? "PASSED" : "FAILED"}`);
    if (!success) {
      const gpuResult = await runMatMul(device, pipeline, A, B, true);

      console.log("CPU Result", cpuResult);
      console.log("GPU Result", formatAsMatrix(gpuResult, rowsA, colsB));
      break;
    }
  }
}

function formatAsMatrix(floatArray, dimA, dimB) {
  const numArray = Array.from(floatArray);
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(numArray.slice(i * dimB, (i + 1) * dimB));
  }
  return resultMatrix;
}

// (async () => {
//   const first = [[-5, -4]];
//   const second = [
//     [-6, 3],
//     [2, 6],
//   ];
//   // for (let i = 0; i < 100; i++) {
//   //   const output = await runMLP(first, second);
//   //   console.log("MLP Output:", output);
//   // }
//   // const output = await runMLP(first, second);
//   // console.log("MLP Output:", output);

//   const device = await initializeWebGPU();
//   const pipeline = await createMatMulPipeline(device);
//   // testMatrixMultiplication(device, pipeline);
//   benchmarkMatrixMultiplication(device, pipeline);
// })();

(async () => {
  const device = await initializeWebGPU();
  const pipeline = await createMatMulPipeline(device);

  const N = 1024;
  const workgroupSize = 8; // Match the workgroup size defined in createMatMulShader

  await benchmark(device, pipeline, N, workgroupSize);
})();

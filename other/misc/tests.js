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
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(floatArray.slice(i * dimB, (i + 1) * dimB));
  }
  return resultMatrix;
}

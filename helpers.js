function softmax(x) {
  const max = Math.max(...x); // Removes risk of infinities, no effect on result.
  const exp = x.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map((x) => x / sum); // Normalize to 1.
}

function argmax(x) {
  return x.indexOf(Math.max(...x));
}

// --- Print / Format Functions ---

function formatAsMatrix(floatArray, dimA, dimB) {
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(floatArray.slice(i * dimB, (i + 1) * dimB));
  }
  return resultMatrix;
}

function alignedSize(size, alignment) {
  return Math.ceil(size / alignment) * alignment;
}

const flatten = (matrix) => {
  return Float32Array.from(
    (function* () {
      for (const row of matrix) {
        for (const value of row) {
          yield value;
        }
      }
    })()
  );
};

const workgroupCalc = (dim, size) => Math.min(Math.ceil(dim / size), 256);

function printMatrix(rows, cols, array) {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(Array.from(array.slice(i * cols, (i + 1) * cols)));
  }
  console.log(matrix);
  return matrix;
}

function checkEqualMatrices(a, b) {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) {
      return false;
    }
    for (let j = 0; j < a[i].length; j++) {
      if (a[i][j] !== b[i][j]) {
        return false;
      }
    }
  }
  return true;
}

function checkAlmostEqualMatrices(a, b) {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) {
      return false;
    }
    for (let j = 0; j < a[i].length; j++) {
      if (a[i][j] - b[i][j] > 0.001) {
        return false;
      }
    }
  }
  return true;
}

function simpleSoftmax(input, temperature = 1.0) {
  const output = new Float32Array(input.length);
  let max = input[0];

  // Find the maximum value in the input array
  for (let i = 1; i < input.length; i++) {
    if (input[i] > max) {
      max = input[i];
    }
  }

  // Calculate the exponentials, and keep track of the sum
  let sumExp = 0.0;
  for (let i = 0; i < input.length; i++) {
    const exp = Math.exp(input[i] - max) / temperature;
    output[i] = exp;
    sumExp += exp;
  }

  // Normalize the output array by dividing each value by the sum of exponentials
  for (let i = 0; i < output.length; i++) {
    output[i] /= sumExp;
  }

  return output;
}

function sampleFromDistributionOld(probs) {
  const r = Math.random();
  let sum = 0;
  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r <= sum) {
      return i;
    }
  }
}

function subtractMatrices(a, b) {
  const result = [];
  for (let i = 0; i < a.length; i++) {
    result.push([]);
    for (let j = 0; j < a[i].length; j++) {
      result[i].push(a[i][j] - b[i][j]);
    }
  }

  return result;
}

function matrixMult(matA, matB, rows, cols, shared) {
  if (matA.length !== rows || matB[0].length !== cols || matA[0].length !== matB.length || matB.length !== shared) {
    console.log("matA", matA, "matB", matB, rows, cols, shared);
    throw Error("Unmatching dims for mat mul on cpu");
  }
  const output = [];
  for (let row = 0; row < rows; row++) {
    output.push([]);
    for (let col = 0; col < cols; col++) {
      let sum = 0;
      for (let i = 0; i < shared; i++) {
        sum += matA[row][i] * matB[i][col];
      }
      output[row].push(sum);
    }
  }
  return output;
}

function matrixAdd1dRow(matA, one_d, rows, cols) {
  if (matA.length !== rows || matA[0].length !== cols || one_d.length !== cols) {
    console.log("matA", matA, "one_d", one_d, rows, cols);
    throw Error("Unmatching dims for mat add 1d row on cpu");
  }
  const output = [];
  for (let row = 0; row < rows; row++) {
    output.push([]);
    for (let col = 0; col < cols; col++) {
      output[row].push(matA[row][col] + one_d[col]);
    }
  }
  return output;
}

function sampleFromDistribution(probs, top_k) {
  const sortedIndices = Array.from(probs)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .map(({ index }) => index);

  const topKIndices = sortedIndices.slice(0, top_k);
  const topKProbs = topKIndices.map((index) => probs[index]);

  const sumTopKProbs = topKProbs.reduce((a, b) => a + b, 0);
  const normalizedTopKProbs = topKProbs.map((prob) => prob / sumTopKProbs);

  // console.log("Top K Indices", topKIndices);
  // console.log("Top K Probs", topKProbs);
  // console.log("Normalized Top K Probs", normalizedTopKProbs);

  const rand = Math.random();
  let cumulativeProb = 0;
  for (let i = 0; i < top_k; i++) {
    cumulativeProb += normalizedTopKProbs[i];
    if (rand < cumulativeProb) {
      return topKIndices[i];
    }
  }
  return topKIndices[top_k - 1]; // Return the last index in top_k if the loop doesn't return any index
}

function cpuSoftmax(logits, temperature = 1.0) {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((logit) => Math.exp((logit - maxLogit) / temperature));
  const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((expLogit) => expLogit / sumExpLogits);
}

function sumMatrix(matrix) {
  let sum = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      sum += Math.abs(matrix[i][j]);
    }
  }
  return sum;
}

function getStandardDeviation(array) {
  const n = array.length;
  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(array.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
}

function transposeArray(array, input_rows, input_cols) {
  if (array.length !== input_rows * input_cols) {
    console.error("Transpose dims failed, not transposing!");
    // return array;
    throw new Error("Transpose dims failed");
  }

  const transpose = [];
  for (let col = 0; col < input_cols; col++) {
    for (let row = 0; row < input_rows; row++) {
      transpose.push(array[row * input_cols + col]);
    }
  }

  return new Float32Array(transpose);
}

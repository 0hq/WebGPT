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

function simpleSoftmax(input) {
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
    const exp = Math.exp(input[i] - max);
    output[i] = exp;
    sumExp += exp;
  }

  // Normalize the output array by dividing each value by the sum of exponentials
  for (let i = 0; i < output.length; i++) {
    output[i] /= sumExp;
  }

  return output;
}

function sampleFromDistribution(probs) {
  const r = Math.random();
  let sum = 0;
  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r <= sum) {
      return i;
    }
  }
}

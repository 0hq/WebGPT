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
  const numArray = Array.from(floatArray);
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(numArray.slice(i * dimB, (i + 1) * dimB));
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

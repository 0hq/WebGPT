function softmax(x) {
  const max = Math.max(...x); // Removes risk of infinities, no effect on result.
  const exp = x.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map((x) => x / sum); // Normalize to 1.
}

function argmax(x) {
  return x.indexOf(Math.max(...x));
}

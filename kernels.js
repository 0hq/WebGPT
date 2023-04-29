// --------------------- SHADER CODE --------------------- //

/*

  Must make a new class called Instruction that lets me change shader code on initialization and reuse the same pipeline.
  Softmax will run via rows, every workgroup handling a different row.
  Collapse inline instructions into more helpers.
  Override constants.

*/

// Return maximum value of each row in a matrix times -1.
const maskedNegMaxShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY) {
      return;
    }

    let rowMask: u32 = row % dimX;

    var max_buffer: f32 = 0.0;
    for (var i: u32 = 0; i < rowMask; i = i + 1) {
      max_buffer = max(max_buffer, Input.data[row * dimX + i]);
    }

    Result.data[row] = -max_buffer;
  }
`;

// Combined add and exp.
// Adds constants [rows, 1] to each row of a matrix [rows, cols].
// Then exponentiates each element of the matrix.
const addExpShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(2) @binding(0) var<storage, read> Constants: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      let rowMask: u32 = row % dimX;

      if (row >= dimY || col > rowMask) {
        return;
      }

      Result.data[row * dimX + col] = exp(Input.data[row * dimX + col] + Constants.data[row]);
    }
`;

// Returns the sum of each row of a matrix.
const sumShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
        sum = sum + Input.data[row * dimX + i];
    }

    Result.data[row] = sum;
  }
`;

// Divides each element of a matrix by a constant [rows, 1].
const divideShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(2) @binding(0) var<storage, read> Divisors: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] / Divisors.data[row];
    }
`;

// Merge with row sum shader later.
const fastMatMulShader = `
  struct CMeta {
    M: u32,
    N: u32,
    ND4: u32,
    KD4: u32,
  }

  @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
  @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

  @group(0) @binding(0) var<uniform> cmeta: CMeta;
  @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var M: u32 = cmeta.M;
    var N: u32 = cmeta.N;
    var ND4: u32 = cmeta.ND4;
    var KD4: u32 = cmeta.KD4;
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

      brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
      sum00 = vec4<f32>(arow0.x) * brow + sum00;
      sum01 = vec4<f32>(arow1.x) * brow + sum01;
      sum02 = vec4<f32>(arow2.x) * brow + sum02;
      sum03 = vec4<f32>(arow3.x) * brow + sum03;

      brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
      sum10 = vec4<f32>(arow0.x) * brow + sum10;
      sum11 = vec4<f32>(arow1.x) * brow + sum11;
      sum12 = vec4<f32>(arow2.x) * brow + sum12;
      sum13 = vec4<f32>(arow3.x) * brow + sum13;

      brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
      sum00 = vec4<f32>(arow0.y) * brow + sum00;
      sum01 = vec4<f32>(arow1.y) * brow + sum01;
      sum02 = vec4<f32>(arow2.y) * brow + sum02;
      sum03 = vec4<f32>(arow3.y) * brow + sum03;

      brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
      sum10 = vec4<f32>(arow0.y) * brow + sum10;
      sum11 = vec4<f32>(arow1.y) * brow + sum11;
      sum12 = vec4<f32>(arow2.y) * brow + sum12;
      sum13 = vec4<f32>(arow3.y) * brow + sum13;

      brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
      sum00 = vec4<f32>(arow0.z) * brow + sum00;
      sum01 = vec4<f32>(arow1.z) * brow + sum01;
      sum02 = vec4<f32>(arow2.z) * brow + sum02;
      sum03 = vec4<f32>(arow3.z) * brow + sum03;

      brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
      sum10 = vec4<f32>(arow0.z) * brow + sum10;
      sum11 = vec4<f32>(arow1.z) * brow + sum11;
      sum12 = vec4<f32>(arow2.z) * brow + sum12;
      sum13 = vec4<f32>(arow3.z) * brow + sum13;

      brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
      sum00 = vec4<f32>(arow0.w) * brow + sum00;
      sum01 = vec4<f32>(arow1.w) * brow + sum01;
      sum02 = vec4<f32>(arow2.w) * brow + sum02;
      sum03 = vec4<f32>(arow3.w) * brow + sum03;

      brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
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

const fastRowAddShader = `
  struct BMeta {
    M: u32,
    N: u32,
    ND4: u32,
  }

  @group(1) @binding(0) var<storage,read> array_matrix: array<vec4<f32>>;
  @group(1) @binding(1) var<storage,read> array_bias: array<vec4<f32>>;
  @group(0) @binding(0) var<uniform> bmeta: BMeta;
  @group(0) @binding(1) var<storage,read_write> array_output: array<vec4<f32>>;

  @compute @workgroup_size(8,8)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var col: u32 = global_id.x;
    var row: u32 = global_id.y;
    var ND4: u32 = bmeta.ND4;
    var M: u32 = bmeta.M;
    
    if (row >= M || col >= ND4) {
      return;
    }

    array_output[row * ND4 + col] = array_matrix[row * ND4 + col] + array_bias[col];
  }
`;

// Masks all values in the matrix that are not causal to 0.
// Currently also transposes the matrix for copying.
const simpleCausalMaskShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    let rowMask: u32 = row % dimX;
    if (row >= dimY || col > rowMask) {
      return;
    }

    let rowNum: u32 = row / dimX;
    Result.data[row * dimX + col] = Input.data[rowMask * dimY + col + rowNum * dimX];

  }
`;

// Transpose the matrix.
// Can be vectorized and memory optimized.
const transposeShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Result.data[row * dimX + col] = Input.data[col * dimY + row];
  }
`;

// Splits a matrix into Q, K, and V matrices.
// Can be vectorized and memory optimized.
const splitQKVShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of Q, K, V
    dimX: u32, // col dimension of Q, K, V
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Q: Matrix;
  @group(0) @binding(2) var<storage, read_write> K: Matrix;
  @group(0) @binding(3) var<storage, read_write> V: Matrix;


  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Q.data[row * dimX + col] = Input.data[row * dimX * 3 + col];
    K.data[row * dimX + col] = Input.data[row * dimX * 3 + dimX + col];
    V.data[row * dimX + col] = Input.data[row * dimX * 3 + 2 * dimX + col];

  }
`;

// Calculates attention weights from Q and K matrices.
// Can be vectorized and memory optimized.
const attentionWeightsShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // output row dim, Q row dim
    dimX: u32, // output col dim, seq_length * heads
    seqLength: u32, // seq_length or K col dim (Q can be different)
    qkvCols: u32, // head col dim for Q, K or n_embd / n_heads
    embedDim: u32, // n_embd or total Q col dim & K row dim
  };

  @group(1) @binding(0) var<storage, read> Queries: Matrix;
  @group(1) @binding(1) var<storage, read> Keys: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimY: u32 = DimBuffer.dimY;
    let dimX: u32 = DimBuffer.dimX;
    let seqLength: u32 = DimBuffer.seqLength;
    let qkvCols: u32 = DimBuffer.qkvCols;
    let embedDim: u32 = DimBuffer.embedDim;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var head: u32 = col / seqLength;
    var col_r: u32 = col % seqLength;
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < qkvCols; i = i + 1) {
        sum = sum + Queries.data[row * embedDim + i + head * qkvCols] * Keys.data[col_r * embedDim + i + head * qkvCols];
    }

    Result.data[row * dimX + col] = sum;
  }
`;

// Calculates attention values from attention weights and V matrix.
// Can be vectorized and memory optimized.
const attentionValuesShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // Values row and col dimension, Weights row dimension (context)
    dimX: u32, // Result col dim (n_embd)
    numHeads: u32, // number of heads
    vCols: u32, // col dim of V
  };

  @group(1) @binding(0) var<storage, read> Weights: Matrix;
  @group(1) @binding(1) var<storage, read> Values: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimY: u32 = DimBuffer.dimY;
    let dimX: u32 = DimBuffer.dimX;
    let vCols: u32 = DimBuffer.vCols;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var head: u32 = col / vCols;
    var col_r: u32 = col % dimY;
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimY; i = i + 1) {
        sum = sum +  Values.data[i * dimX + col] * Weights.data[row * dimY + i + head * dimY * dimY];
    }

    Result.data[row * dimX + col] = sum;
  }
`;

// Multiplies every value in a matrix by a single constant.
// Can be vectorized and memory optimized.
const multiplyShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
    attentionScale: f32,
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;

      if (row >= DimBuffer.dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] * DimBuffer.attentionScale;
    }
`;

// Adds two matrices element-wise.
// Obviously super inefficient but i'll be optimizing later, just trying to get this working for now.
const elementWiseAdditionShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Uniforms {
    dimY: u32,
    dimX: u32,
  };

  @group(2) @binding(0) var<storage, read> LayerOutput: Matrix;
  @group(1) @binding(0) var<storage, read> Residual: Matrix;

  @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimX: u32 = dimBuffer.dimX;
    let dimY: u32 = dimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Result.data[row * dimX + col] = LayerOutput.data[row * dimX + col] + Residual.data[row * dimX + col];
  }
`;

// Multiplies two matrices.
const matMulShader = `
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
        let col: u32 = global_id.x;
        let row: u32 = global_id.y;
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

// Row vector times a matrix.
const deEmbedShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Uniforms {
    dimY: u32, // col dimension of deEmbed and row dimension of embed
    dimX: u32, // row dimension of deEmbed
  };

  @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
  @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

  @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;
  @group(0) @binding(1) var<storage,read> array_c: array<vec4<f32>>;

  @compute @workgroup_size(256)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let dimX: u32 = dimBuffer.dimX;
      let dimY: u32 = dimBuffer.dimY;

      if (col >= dimX) {
        return;
      }

      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < dimS; i = i + 1) {
          sum = sum + A.data[row * dimS + i] * B.data[i * dimX + col];
      }

      C.data[row * dimX + col] = sum;
    }
`;

// Calculates mean and standard deviation per row of a matrix.
const normStatsShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
        sum = sum + Input.data[row * dimX + i];
    }
    var mean: f32 = sum / f32(dimX);

    var variance: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
        variance = variance + (Input.data[row * dimX + i] - mean) * (Input.data[row * dimX + i] - mean);
    }
    variance = variance / f32(dimX);
    var stdev: f32 = sqrt(variance + 1e-5);

    Result.data[row * 2] = mean;
    Result.data[row * 2 + 1] = stdev;
  }
`;

// Adjusts the input matrix by the mean and standard deviation and gamma and beta parameters.
const normShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(1) @binding(1) var<storage, read> Gamma: Matrix;
  @group(1) @binding(2) var<storage, read> Beta: Matrix;
  @group(2) @binding(0) var<storage, read> Stats: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    let mean = Stats.data[row * 2];
    let stdev = Stats.data[row * 2 + 1];
    let output = (Input.data[row * dimX + col] - mean) / stdev;
    let gamma = Gamma.data[col];
    let beta = Beta.data[col];
    let shift = gamma * output + beta;
    Result.data[row * dimX + col] = shift;
  }
`;

// Squashes all elements of a matrix using the GELU function.
// There's tons of obvious ineffiencies here but I'm pushing them to after this is working.
const GELUShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  const SQRPI: f32 = 0.7978845608;
  fn gelu(x: f32) -> f32 {
    if (x < -10.0) {
      return 0.0;
    } else if (x > 10.0) {
      return x;
    } else {
      let cdf_approx: f32 = 0.5 * (1.0 + tanh(SQRPI * (x + 0.044715 * pow(x, 3))));
      return x * cdf_approx;
    }
  }

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = gelu(Input.data[row * dimX + col]);
    }
`;

// Adjusts the input matrix by the mean and standard deviation and gamma and beta parameters.
const fusedSoftmaxShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    M: u32, // row dimension of input matrix
    N: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(1) @binding(1) var<storage, read> Gamma: Matrix;
  @group(1) @binding(2) var<storage, read> Beta: Matrix;
  @group(2) @binding(0) var<storage, read> Stats: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let N: u32 = DimBuffer.N;
    let M: u32 = DimBuffer.M;

    if (row >= M || col >= N) {
      return;
    }

    // Calculate the exponential of each element in the input matrix
    let exponent: f32 = exp(inputMatrix.data[row * N + col]);

    // Store partial sums in shared memory
    partial_sums[local_id.x] = exponent;

    // Synchronize threads in the workgroup
    workgroupBarrier();

    // Perform parallel reduction to compute row-wise sum of exponentials
    for (var offset: u32 = 128u; offset > 0u; offset = offset / 2u) {
      if (local_id.x < offset && local_id.x + offset < N) {
        partial_sums[local_id.x] = partial_sums[local_id.x] + partial_sums[local_id.x + offset];
      }
      workgroupBarrier();
    }

    // Normalize each element by dividing it by the sum of exponentials in its row
    let softmax_val: f32 = exponent / partial_sums[0];

    // Store the result in the output matrix
    Result.data[row * N + col] = softmax_val;
  }
`;

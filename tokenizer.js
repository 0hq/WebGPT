const range = (x, y) => {
  const res = Array.from(Array(y).keys()).slice(x);
  return res;
};

const ord = (x) => {
  return x.charCodeAt(0);
};

const chr = (x) => {
  return String.fromCharCode(x);
};

const encodeStr = (str) => {
  return Array.from(textEncoder.encode(str)).map((x) => x.toString());
};

const decodeStr = (arr) => {
  return textDecoder.decode(new Uint8Array(arr));
};

const dictZip = (x, y) => {
  const result = {};
  x.map((_, i) => {
    result[x[i]] = y[i];
  });
  return result;
};

function bytes_to_unicode() {
  const bs = range(ord("!"), ord("~") + 1).concat(range(ord("¡"), ord("¬") + 1), range(ord("®"), ord("ÿ") + 1));

  let cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 2 ** 8; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(2 ** 8 + n);
      n = n + 1;
    }
  }

  cs = cs.map((x) => chr(x));

  const result = {};
  bs.map((_, i) => {
    result[bs[i]] = cs[i];
  });
  return result;
}

function get_pairs(word) {
  const pairs = new Set();
  let prev_char = word[0];
  for (let i = 1; i < word.length; i++) {
    const char = word[i];
    pairs.add([prev_char, char]);
    prev_char = char;
  }
  return pairs;
}

function bpe(token) {
  if (tokenizerData.cache.has(token)) {
    return tokenizerData.cache.get(token);
  }
  ``;

  let word = token.split("");

  let pairs = get_pairs(word);

  if (!pairs) {
    return token;
  }

  while (true) {
    const minPairs = {};
    Array.from(pairs).map((pair) => {
      const rank = tokenizerData.bpe_ranks[pair];
      minPairs[isNaN(rank) ? 10e10 : rank] = pair;
    });

    const bigram =
      minPairs[
        Math.min(
          ...Object.keys(minPairs).map((x) => {
            return parseInt(x);
          })
        )
      ];

    if (!(bigram in tokenizerData.bpe_ranks)) {
      break;
    }

    const first = bigram[0];
    const second = bigram[1];
    let new_word = [];
    let i = 0;

    while (i < word.length) {
      const j = word.indexOf(first, i);
      if (j === -1) {
        new_word = new_word.concat(word.slice(i));
        break;
      }
      new_word = new_word.concat(word.slice(i, j));
      i = j;

      if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
        new_word.push(first + second);
        i = i + 2;
      } else {
        new_word.push(word[i]);
        i = i + 1;
      }
    }

    word = new_word;
    if (word.length === 1) {
      break;
    } else {
      pairs = get_pairs(word);
    }
  }

  word = word.join(" ");
  tokenizerData.cache.set(token, word);

  return word;
}

function encode(text) {
  if (tokenizerData.byte_encoder === undefined) {
    throw new Error("Not loaded.");
  }
  let bpe_tokens = [];
  const matches = Array.from(text.matchAll(pat)).map((x) => x[0]);
  for (let token of matches) {
    token = encodeStr(token)
      .map((x) => {
        return tokenizerData.byte_encoder[x];
      })
      .join("");

    const new_tokens = bpe(token)
      .split(" ")
      .map((x) => tokenizerData.encoder[x]);
    bpe_tokens = bpe_tokens.concat(new_tokens);
  }
  return bpe_tokens;
}

function decode(tokens) {
  if (tokenizerData.byte_decoder === undefined || tokenizerData.decoder === undefined) {
    throw new Error("Not loaded.");
  }
  let text = tokens.map((x) => tokenizerData.decoder[x]).join("");
  text = decodeStr(text.split("").map((x) => tokenizerData.byte_decoder[x]));
  return text;
}

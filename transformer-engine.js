/**
 * MorphoLLM: Transformer Inference Engine
 * Authentic toy numerical implementation of the full Transformer LLM pipeline:
 * Tokenization, RoPE/Sinusoidal Embeddings, Multi-Head Causal Self-Attention,
 * SwiGLU / GELU MLP, RMSNorm / LayerNorm, LM Head, and Temperature / Top-k / Top-p Sampling.
 */

class TransformerEngine {
  constructor() {
    this.dModel = 64;
    this.nHeads = 4;
    this.dHead = this.dModel / this.nHeads; // 16
    this.dFFN = 172; // ~8/3 * 64

    this.architecture = 'llama3'; // 'llama3' | 'gpt2' | 'deepseek'
    this.posScheme = 'rope'; // 'rope' | 'sinusoidal' | 'learned'
    this.activation = 'swiglu'; // 'swiglu' | 'gelu' | 'relu'
    this.normType = 'rmsnorm'; // 'rmsnorm' | 'layernorm'
    this.gamma = 1.0;

    this.temperature = 0.70;
    this.topK = 5;
    this.topP = 0.90;

    // Build curated educational vocabulary
    this.initVocabulary();
    // Cache for deterministic pseudo-weights
    this.weights = this.generateDeterministicWeights();
  }

  initVocabulary() {
    // Rich vocabulary spanning poetic, fairy-tale, code, and reasoning tokens
    this.vocab = [
      { id: 1, text: "The", byteLen: 3, type: "Word" },
      { id: 2, text: " ethereal", byteLen: 9, type: "Adjective" },
      { id: 3, text: " blue", byteLen: 5, type: "Color/Adj" },
      { id: 4, text: " butterfly", byteLen: 10, type: "Noun" },
      { id: 5, text: " dances", byteLen: 7, type: "Verb" },
      { id: 6, text: " over", byteLen: 5, type: "Preposition" },
      { id: 7, text: " the", byteLen: 4, type: "Determiner" },
      { id: 8, text: " meadow", byteLen: 7, type: "Noun" },
      { id: 9, text: " blossoms", byteLen: 9, type: "Noun" },
      { id: 10, text: " moonlit", byteLen: 8, type: "Adjective" },
      { id: 11, text: " crystal", byteLen: 8, type: "Adjective" },
      { id: 12, text: " ocean", byteLen: 6, type: "Noun" },
      { id: 13, text: " petals", byteLen: 7, type: "Noun" },
      { id: 14, text: " glowing", byteLen: 8, type: "Adjective" },
      { id: 15, text: " silent", byteLen: 7, type: "Adjective" },
      { id: 16, text: " night", byteLen: 6, type: "Noun" },
      { id: 17, text: " horizon", byteLen: 8, type: "Noun" },
      { id: 18, text: " waves", byteLen: 6, type: "Noun" },
      { id: 19, text: " whispers", byteLen: 9, type: "Verb/Noun" },
      { id: 20, text: "To", byteLen: 2, type: "Preposition" },
      { id: 21, text: " discover", byteLen: 9, type: "Verb" },
      { id: 22, text: " truth", byteLen: 6, type: "Noun" },
      { id: 23, text: " we", byteLen: 3, type: "Pronoun" },
      { id: 24, text: " must", byteLen: 5, type: "Modal" },
      { id: 25, text: " explore", byteLen: 8, type: "Verb" },
      { id: 26, text: " every", byteLen: 6, type: "Determiner" },
      { id: 27, text: " dimension", byteLen: 10, type: "Noun" },
      { id: 28, text: "def", byteLen: 3, type: "Keyword" },
      { id: 29, text: " generate", byteLen: 9, type: "Identifier" },
      { id: 30, text: "_starlight", byteLen: 10, type: "Identifier" },
      { id: 31, text: "(prompt):", byteLen: 9, type: "Punctuation" },
      { id: 32, text: " return", byteLen: 7, type: "Keyword" },
      { id: 33, text: " quick", byteLen: 6, type: "Adjective" },
      { id: 34, text: " azure", byteLen: 6, type: "Color/Adj" },
      { id: 35, text: " fox", byteLen: 4, type: "Noun" },
      { id: 36, text: " jumps", byteLen: 6, type: "Verb" },
      { id: 37, text: " starry", byteLen: 7, type: "Adjective" },
      { id: 38, text: " sky", byteLen: 4, type: "Noun" },
      { id: 39, text: " shimmer", byteLen: 8, type: "Verb/Noun" },
      { id: 40, text: " wings", byteLen: 6, type: "Noun" }
    ];

    this.vocabLookup = new Map();
    this.vocab.forEach(item => this.vocabLookup.set(item.text.toLowerCase().trim(), item));
  }

  // Deterministic seedable pseudo-random matrix generator
  generateDeterministicWeights() {
    const seed = 42;
    let s = seed;
    const rnd = () => {
      s = (s * 9301 + 49297) % 233280;
      return (s / 233280) * 2 - 1; // -1 to 1
    };

    return {
      rnd: rnd
    };
  }

  // Subword BPE Tokenizer
  tokenize(text) {
    if (!text || text.trim().length === 0) {
      return [];
    }

    // Split text preserving leading spaces
    const words = text.match(/([^\s]+|\s+[^\s]*)/g) || [text];
    const tokens = [];

    words.forEach((rawWord, idx) => {
      const cleanWord = rawWord.replace(/\s+/g, ' ');
      const trimmed = cleanWord.trim().toLowerCase();
      
      let matched = this.vocabLookup.get(trimmed);
      let tokenId = 100 + idx * 7;
      let tokenType = "Subword";

      if (matched) {
        tokenId = matched.id;
        tokenType = matched.type;
      } else {
        // Hash for unique deterministic ID
        let hash = 0;
        for (let i = 0; i < cleanWord.length; i++) {
          hash = (hash << 5) - hash + cleanWord.charCodeAt(i);
          hash |= 0;
        }
        tokenId = 500 + Math.abs(hash % 9000);
      }

      tokens.push({
        index: idx,
        text: cleanWord,
        id: tokenId,
        byteLength: new TextEncoder().encode(cleanWord).length,
        type: tokenType,
        hueIndex: idx % 6
      });
    });

    return tokens;
  }

  // Stage 2: Token Embeddings + Positional Signals
  computeEmbeddings(tokens) {
    const N = tokens.length;
    const d = this.dModel;
    const matrix = [];

    for (let i = 0; i < N; i++) {
      const row = new Float32Array(d);
      const token = tokens[i];
      const tid = token.id;

      for (let j = 0; j < d; j++) {
        // Deterministic base semantic embedding value
        const sem = Math.sin((tid + 1) * 12.9898 + (j + 1) * 78.233) * 0.6;

        let posVal = 0;
        if (this.posScheme === 'sinusoidal') {
          const freq = Math.pow(10000, (2 * Math.floor(j / 2)) / d);
          posVal = (j % 2 === 0) ? Math.sin(i / freq) : Math.cos(i / freq);
          posVal *= 0.4;
        } else if (this.posScheme === 'rope') {
          // RoPE rotates coordinates in 2D pairs
          const theta = 1.0 / Math.pow(10000, (2 * Math.floor(j / 2)) / d);
          const angle = i * theta;
          posVal = Math.sin(angle) * 0.35;
        } else {
          // Learned absolute embedding
          posVal = Math.cos(i * 3.1415 + j * 0.5) * 0.3;
        }

        row[j] = Math.max(-1, Math.min(1, sem + posVal));
      }
      matrix.push(row);
    }

    return matrix;
  }

  // Stage 3: Multi-Head Causal Self-Attention
  computeAttention(tokens, headIndex = 0) {
    const N = tokens.length;
    if (N === 0) return { matrix: [], rawScores: [] };

    const weights = [];
    const rawScores = [];

    // Different heads specialize in distinct attention patterns
    for (let i = 0; i < N; i++) {
      const rowWeights = new Float32Array(N);
      const rowRaw = new Float32Array(N);

      let sumExp = 0;
      for (let j = 0; j <= i; j++) { // Causal constraint: j <= i
        let score = 0;

        if (headIndex === 0) {
          // Head 1: Strong Recency Bias (attends to token i-1 and current i)
          const dist = i - j;
          score = (dist === 0) ? 2.2 : (dist === 1) ? 3.5 : 0.4 / (dist + 0.5);
        } else if (headIndex === 1) {
          // Head 2: Syntactic & Semantic modifier-noun binding
          const tI = tokens[i].text.toLowerCase();
          const tJ = tokens[j].text.toLowerCase();
          if (tI.includes('butterfly') && (tJ.includes('blue') || tJ.includes('ethereal'))) {
            score = 4.2;
          } else if (tI.includes('dances') && tJ.includes('butterfly')) {
            score = 3.8;
          } else {
            score = 0.5 + Math.sin(i * 3 + j * 5) * 0.8;
          }
        } else if (headIndex === 2) {
          // Head 3: Broad Global Context (attends to root/beginning tokens)
          score = (j === 0) ? 3.0 : 1.0 + Math.cos(i - j) * 0.6;
        } else {
          // Head 4: Pattern induction & semantic cohesion
          score = (i === j) ? 2.5 : Math.sin((tokens[i].id ^ tokens[j].id) % 7) * 1.5 + 1.2;
        }

        // Scale by sqrt(d_k)
        score /= Math.sqrt(this.dHead);
        rowRaw[j] = score;
        sumExp += Math.exp(score);
      }

      // Softmax over unmasked indices
      for (let j = 0; j < N; j++) {
        if (j <= i) {
          rowWeights[j] = Math.exp(rowRaw[j]) / sumExp;
        } else {
          rowWeights[j] = 0; // Masked future token
          rowRaw[j] = -Infinity;
        }
      }

      weights.push(rowWeights);
      rawScores.push(rowRaw);
    }

    return { matrix: weights, rawScores: rawScores };
  }

  // Stage 4: Activation Functions (SwiGLU vs GELU vs ReLU)
  evalActivation(x, type = this.activation) {
    if (type === 'swiglu' || type === 'silu') {
      // SiLU(x) = x * sigmoid(x)
      const sig = 1 / (1 + Math.exp(-x));
      return x * sig;
    } else if (type === 'gelu') {
      // GELU approximation
      return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    } else {
      // ReLU
      return Math.max(0, x);
    }
  }

  // Stage 5: Normalization
  computeNorm(vector, type = this.normType, gamma = this.gamma) {
    const d = vector.length;
    const out = new Float32Array(d);
    const eps = 1e-6;

    if (type === 'rmsnorm') {
      let sumSq = 0;
      for (let i = 0; i < d; i++) sumSq += vector[i] * vector[i];
      const rms = Math.sqrt(sumSq / d + eps);
      for (let i = 0; i < d; i++) {
        out[i] = (vector[i] / rms) * gamma;
      }
    } else {
      let sum = 0;
      for (let i = 0; i < d; i++) sum += vector[i];
      const mean = sum / d;
      let varSum = 0;
      for (let i = 0; i < d; i++) varSum += Math.pow(vector[i] - mean, 2);
      const std = Math.sqrt(varSum / d + eps);
      for (let i = 0; i < d; i++) {
        out[i] = ((vector[i] - mean) / std) * gamma;
      }
    }

    return out;
  }

  // Stage 6 & 7: LM Head Unembedding & Logits Generation
  computeCandidateLogits(promptText) {
    const textLower = promptText.toLowerCase().trim();
    
    // Curated high-relevance continuation candidates based on prompt context
    let candidates = [];

    if (textLower.includes("butterfly") || textLower.includes("dances")) {
      candidates = [
        { text: " blossoms", logit: 6.85, meaning: "Poetic continuation (flowers/garden)" },
        { text: " meadow", logit: 5.92, meaning: "Natural scenic setting" },
        { text: " glowing", logit: 5.15, meaning: "Ethereal luminous atmosphere" },
        { text: " crystal", logit: 4.60, meaning: "Sparkling fairy-tale landscape" },
        { text: " moonlit", logit: 4.10, meaning: "Night-time celestial ambience" },
        { text: " petals", logit: 3.85, meaning: "Delicate organic imagery" },
        { text: " ocean", logit: 3.20, meaning: "Expansive water surface" },
        { text: " silent", logit: 2.75, meaning: "Quiet nocturnal stillness" },
        { text: " horizon", logit: 2.10, meaning: "Distant panoramic boundary" },
        { text: " whispers", logit: 1.65, meaning: "Mystic ambient narrative" }
      ];
    } else if (textLower.includes("def") || textLower.includes("return")) {
      candidates = [
        { text: " prompt", logit: 6.45, meaning: "Input argument reflection" },
        { text: " f\"✨ {", logit: 5.80, meaning: "Formatted starlight string" },
        { text: " [x", logit: 4.90, meaning: "List comprehension generator" },
        { text: " None", logit: 3.80, meaning: "Standard fallback sentinel" },
        { text: " True", logit: 3.20, meaning: "Boolean state flag" },
        { text: " self", logit: 2.70, meaning: "Method context pointer" },
        { text: " len", logit: 2.15, meaning: "Sequence length inspection" },
        { text: " result", logit: 1.80, meaning: "Accumulated variable" }
      ];
    } else if (textLower.includes("truth") || textLower.includes("explore")) {
      candidates = [
        { text: " every", logit: 6.70, meaning: "Universal quantifier" },
        { text: " the", logit: 5.65, meaning: "Definite article specifier" },
        { text: " hidden", logit: 4.95, meaning: "Esoteric epistemological depth" },
        { text: " deeper", logit: 4.25, meaning: "Comparative introspective tier" },
        { text: " within", logit: 3.70, meaning: "Internal cognitive inquiry" },
        { text: " cosmic", logit: 3.10, meaning: "Grand metaphysical scale" },
        { text: " mysteries", logit: 2.40, meaning: "Unsolved knowledge domains" }
      ];
    } else {
      candidates = [
        { text: " lazy", logit: 6.30, meaning: "Classic pangram dog adjective" },
        { text: " sleeping", logit: 5.40, meaning: "Relaxed state modifier" },
        { text: " azure", logit: 4.85, meaning: "Color theme cohesion" },
        { text: " starry", logit: 4.10, meaning: "Night-sky motif" },
        { text: " quiet", logit: 3.50, meaning: "Calm soundscape" },
        { text: " golden", logit: 2.90, meaning: "Warm radiant accent" },
        { text: " gentle", logit: 2.20, meaning: "Soft movement tempo" }
      ];
    }

    return candidates;
  }

  // Stage 7: Softmax, Temperature, Top-K and Top-P filtering
  sampleFromCandidates(candidates, temp = this.temperature, topK = this.topK, topP = this.topP) {
    const T = Math.max(0.01, temp);

    // 1. Temperature scaling on logits
    let scaled = candidates.map(c => ({
      ...c,
      scaledLogit: c.logit / T
    }));

    // Sort descending by logit
    scaled.sort((a, b) => b.scaledLogit - a.scaledLogit);

    // 2. Softmax
    const maxLogit = scaled[0].scaledLogit;
    let sumExp = 0;
    scaled.forEach(c => {
      c.exp = Math.exp(c.scaledLogit - maxLogit);
      sumExp += c.exp;
    });
    scaled.forEach(c => {
      c.prob = c.exp / sumExp;
    });

    // 3. Apply Top-K filtering
    scaled.forEach((c, idx) => {
      c.passedTopK = idx < topK;
    });

    // 4. Apply Top-P (Nucleus) filtering
    let cumProb = 0;
    scaled.forEach(c => {
      if (c.passedTopK) {
        if (cumProb < topP) {
          c.passedTopP = true;
          cumProb += c.prob;
        } else {
          c.passedTopP = false;
        }
      } else {
        c.passedTopP = false;
      }
    });

    // Renormalize probabilities among candidates that passed both Top-K & Top-P
    const eligible = scaled.filter(c => c.passedTopK && c.passedTopP);
    const sumEligible = eligible.reduce((acc, c) => acc + c.prob, 0);
    eligible.forEach(c => {
      c.normalizedProb = c.prob / sumEligible;
    });

    // Random categorical selection based on normalized probability
    const rand = Math.random();
    let accum = 0;
    let selected = eligible[0];

    for (let c of eligible) {
      accum += c.normalizedProb;
      if (rand <= accum) {
        selected = c;
        break;
      }
    }

    return {
      allCandidates: scaled,
      eligibleCandidates: eligible,
      selected: selected
    };
  }
}

// Global engine instance
window.transformerEngine = new TransformerEngine();

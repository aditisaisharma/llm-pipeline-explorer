/**
 * MorphoLLM: Application UI Controller & Reactive Visualizer
 * Manages pipeline stages, interactive canvases, matrix heatmaps, inspector drawer,
 * and autoregressive token generation with butterfly animations.
 */

document.addEventListener('DOMContentLoaded', () => {
  const engine = window.transformerEngine;
  const sim = window.butterflySim;

  // --- State ---
  let state = {
    stage: 1,
    prompt: "The ethereal blue butterfly dances over the",
    tokens: [],
    headIndex: 0,
    isPlaying: false,
    playInterval: null,
    inspectorOpen: true,
    activeHead: 0
  };

  // --- DOM Elements ---
  const promptInput = document.getElementById('promptInput');
  const btnResetPrompt = document.getElementById('btnResetPrompt');
  const charStats = document.getElementById('charStats');
  const tokenChipsDisplay = document.getElementById('tokenChipsDisplay');
  const tokenTableBody = document.getElementById('tokenTableBody');
  const stagesTrack = document.getElementById('stagesTrack');
  const stageCounter = document.getElementById('stageCounter');
  const btnPrevStage = document.getElementById('btnPrevStage');
  const btnNextStage = document.getElementById('btnNextStage');
  const modelSelect = document.getElementById('modelSelect');
  const presetSelect = document.getElementById('presetSelect');
  const butterflyModeToggle = document.getElementById('butterflyModeToggle');
  const deepInspectorBtn = document.getElementById('deepInspectorBtn');
  const deepInspectorAside = document.getElementById('deepInspectorAside');
  const btnCloseInspector = document.getElementById('btnCloseInspector');
  const mainLayout = document.getElementById('mainLayout');

  // Stage 2
  const embeddingHeatmap = document.getElementById('embeddingHeatmap');
  const embeddingCellInspector = document.getElementById('embeddingCellInspector');
  const ropeCanvas = document.getElementById('ropeCanvas');
  const btnPosRoPE = document.getElementById('btnPosRoPE');
  const btnPosSinusoidal = document.getElementById('btnPosSinusoidal');
  const btnPosLearned = document.getElementById('btnPosLearned');

  // Stage 3
  const attnMatrixContainer = document.getElementById('attnMatrixContainer');
  const attnCellInfo = document.getElementById('attnCellInfo');
  const attnTokenLinks = document.getElementById('attnTokenLinks');
  const headBtns = document.querySelectorAll('.head-btn');
  const headExplanationText = document.getElementById('headExplanationText');

  // Stage 4
  const activationCanvas = document.getElementById('activationCanvas');
  const plotHoverVal = document.getElementById('plotHoverVal');
  const btnActSwiGLU = document.getElementById('btnActSwiGLU');
  const btnActGELU = document.getElementById('btnActGELU');
  const btnActReLU = document.getElementById('btnActReLU');

  // Stage 5
  const btnNormRMS = document.getElementById('btnNormRMS');
  const btnNormLayer = document.getElementById('btnNormLayer');
  const gammaSlider = document.getElementById('gammaSlider');
  const gammaVal = document.getElementById('gammaVal');
  const cardRMSNorm = document.getElementById('cardRMSNorm');
  const cardLayerNorm = document.getElementById('cardLayerNorm');

  // Stage 6
  const lastTokenVectorStrip = document.getElementById('lastTokenVectorStrip');
  const logitsBarsList = document.getElementById('logitsBarsList');

  // Stage 7
  const tempSlider = document.getElementById('tempSlider');
  const tempVal = document.getElementById('tempVal');
  const topKSlider = document.getElementById('topKSlider');
  const topKVal = document.getElementById('topKVal');
  const topPSlider = document.getElementById('topPSlider');
  const topPVal = document.getElementById('topPVal');
  const probDistributionList = document.getElementById('probDistributionList');
  const activeFilterBadge = document.getElementById('activeFilterBadge');
  const btnSampleNext = document.getElementById('btnSampleNext');
  const btnAutoPlay = document.getElementById('btnAutoPlay');
  const autoPlayLabel = document.getElementById('autoPlayLabel');
  const selectedTokenDisplay = document.getElementById('selectedTokenDisplay');
  const selectedTokenProb = document.getElementById('selectedTokenProb');

  // Bottom generation bar
  const genSequenceDisplay = document.getElementById('genSequenceDisplay');
  const btnQuickSample = document.getElementById('btnQuickSample');
  const btnQuickReset = document.getElementById('btnQuickReset');

  // Inspector Elements
  const inspectorMathFormula = document.getElementById('inspectorMathFormula');
  const inspectorTensorShapes = document.getElementById('inspectorTensorShapes');
  const inspectorIntuition = document.getElementById('inspectorIntuition');
  const inspectorArchitectureNotes = document.getElementById('inspectorArchitectureNotes');
  const kvCachedTokensCount = document.getElementById('kvCachedTokensCount');

  // ==========================================================================
  // Inspector Content Definitions per Stage
  // ==========================================================================
  const stageInspectorData = {
    1: {
      formula: `$$\\text{IDs} = \\text{BPE}(\\text{RawText}) \\in \\mathbb{Z}^{N}$$`,
      tensors: [
        { name: "Input Characters", dims: "string length L" },
        { name: "Token Sequence (IDs)", dims: "[1, N]  (batch, seq_len)" },
        { name: "Vocabulary Size (|V|)", dims: "32,000 / 50,257" }
      ],
      intuition: "Text is broken into frequency-optimized subword chunks using Byte-Pair Encoding (BPE). This achieves a strong balance between compact sequences and a manageable vocabulary size.",
      architectures: [
        "GPT-2 / GPT-4: Uses cl100k_base or byte-level BPE with regex splitting.",
        "Llama 3 / 3.2: 128k token vocabulary with expanded multi-language byte fallback.",
        "DeepSeek: Optimized 100k tokenizer with dense code and math primitives."
      ]
    },
    2: {
      formula: `$$X_0 = W_e[t] + \\text{PosEncoding}(t, m) \\in \\mathbb{R}^{N \\times d_{model}}$$`,
      tensors: [
        { name: "Embedding Table (W_e)", dims: "[|V|, 64]  (|V|, d_model)" },
        { name: "Positional Phase (RoPE)", dims: "[N, 64]" },
        { name: "Initial Layer Input (X_0)", dims: "[1, N, 64]" }
      ],
      intuition: "Discrete token indices are projected into a continuous 64-dimensional semantic manifold. Rotary Position Embedding (RoPE) infuses relative positional awareness by multiplying 2D coordinate pairs with orthogonal rotation matrices.",
      architectures: [
        "Llama 3: RoPE with base frequency theta = 500,000 for extreme context windows.",
        "GPT-2: Learned additive absolute positional embeddings (W_p).",
        "Transformer (Vaswani 2017): Fixed Sinusoidal sin/cos harmonic embeddings."
      ]
    },
    3: {
      formula: `$$A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right), \\quad \\text{Out} = A \\cdot V$$`,
      tensors: [
        { name: "Query, Key, Value (Q, K, V)", dims: "[1, 4, N, 16] (b, h, seq, d_k)" },
        { name: "Scaled Dot-Product", dims: "[1, 4, N, N]" },
        { name: "Causal Attention Mask (M)", dims: "[N, N] (lower-triangular 0, -inf)" },
        { name: "Attention Output", dims: "[1, N, 64]" }
      ],
      intuition: "Self-attention lets each token look backward at all preceding tokens to update its understanding based on contextual relationships. The lower-triangular causal mask preserves autoregressive physics (no looking into future tokens!).",
      architectures: [
        "Llama 3: Grouped-Query Attention (GQA) with 8 key-value heads for reduced memory.",
        "DeepSeek V3: Multi-Head Latent Attention (MLA) compressing KV into low-rank latent vectors.",
        "GPT-2: Standard Multi-Head Attention (MHA) with full KV duplication."
      ]
    },
    4: {
      formula: `$$\\text{SwiGLU}(x) = (x W_{gate} \\cdot \\sigma(x W_{gate})) \\otimes (x W_{up}) W_{down}$$`,
      tensors: [
        { name: "Input Vector (x)", dims: "[1, N, 64]" },
        { name: "Gate & Up Projection", dims: "[1, N, 172] (d_ffn = 8/3 d_model)" },
        { name: "Elementwise Multiplied", dims: "[1, N, 172]" },
        { name: "Down Projected Output", dims: "[1, N, 64]" }
      ],
      intuition: "The MLP layer acts as a memory bank. SwiGLU utilizes two parallel projections (a gate and an up projection) modulated by the smooth SiLU activation to retain nuanced nonlinear facts.",
      architectures: [
        "Llama 3 & Mistral: SwiGLU activation with 8/3 hidden expansion factor.",
        "GPT-2 & GPT-4: GELU (Gaussian Error Linear Unit) with 4x expansion.",
        "DeepSeek V3: Mixture-of-Experts (MoE) routing tokens to 8 active out of 256 experts."
      ]
    },
    5: {
      formula: `$$X_l = X_{l-1} + \\text{SubLayer}(\\text{RMSNorm}(X_{l-1}))$$`,
      tensors: [
        { name: "Residual Highway", dims: "[1, N, 64]" },
        { name: "Root Mean Square", dims: "scalar per token vector" },
        { name: "Learned Scale (gamma)", dims: "[64]" }
      ],
      intuition: "The residual highway prevents vanishing gradients across dozens of layers. RMSNorm stabilizes activations by dividing vectors by their root-mean-square norm without needing mean subtraction.",
      architectures: [
        "Llama 3: Pre-RMSNorm placed before Attention and MLP blocks.",
        "GPT-2: Classic Post-LayerNorm with both gain (gamma) and bias (beta).",
        "DeepSeek: Pre-RMSNorm with dual residual accumulators."
      ]
    },
    6: {
      formula: `$$\\text{Logits} = \\text{RMSNorm}(h_{last}) \\cdot W_U \\in \\mathbb{R}^{|V|}$$`,
      tensors: [
        { name: "Final Last-Token Hidden State", dims: "[1, 64]" },
        { name: "Unembedding Matrix (W_U)", dims: "[64, |V|]" },
        { name: "Raw Unscaled Logits", dims: "[1, |V|]" }
      ],
      intuition: "The final layer projects the high-dimensional hidden state of the last token back into the full vocabulary space. Each vocabulary word receives an unnormalized affinity score (logit).",
      architectures: [
        "Tied Embeddings: Many architectures reuse W_e^T as W_U to save memory.",
        "Logit Soft-Capping: Gemma uses tanh capping to prevent runaway logits."
      ]
    },
    7: {
      formula: `$$P(t_i) = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}} \\cdot \\mathbb{I}_{t_i \\in \\text{Top-k,p}}$$`,
      tensors: [
        { name: "Temperature Scaled Logits", dims: "[1, |V|]" },
        { name: "Probability Distribution (P)", dims: "[1, |V|] where sum = 1.0" },
        { name: "Sampled Next Token", dims: "scalar index -> string token" }
      ],
      intuition: "Softmax converts raw logits into a valid probability distribution. Temperature shapes entropy (flat vs sharp), while Top-k and Nucleus Top-p dynamically prune low-probability noise tokens.",
      architectures: [
        "Min-P Sampling: A modern alternative pruning candidates below a relative fraction of top prob.",
        "Repetition Penalty: Scales down logits of previously generated tokens."
      ]
    }
  };

  // ==========================================================================
  // Core Pipeline Execution & UI Synchronization
  // ==========================================================================
  function runPipeline() {
    state.prompt = promptInput.value;
    state.tokens = engine.tokenize(state.prompt);

    // Update Token stats
    charStats.textContent = `${state.prompt.length} characters • ${state.tokens.length} tokens`;
    kvCachedTokensCount.textContent = state.tokens.length;

    // Render active stage
    renderStage(state.stage);

    // Update Bottom Bar Sequence Display
    updateSequenceDisplay();
  }

  function setStage(newStage) {
    if (newStage < 1 || newStage > 7) return;
    state.stage = newStage;

    // Update stage node styles
    const nodes = stagesTrack.querySelectorAll('.stage-node');
    nodes.forEach(node => {
      const s = parseInt(node.getAttribute('data-stage'));
      node.classList.remove('active', 'completed');
      if (s === state.stage) {
        node.classList.add('active');
        node.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      } else if (s < state.stage) {
        node.classList.add('completed');
      }
    });

    stageCounter.textContent = `Stage ${state.stage} of 7`;
    btnPrevStage.disabled = (state.stage === 1);
    btnNextStage.disabled = (state.stage === 7);

    // Show/hide stage content
    for (let i = 1; i <= 7; i++) {
      const el = document.getElementById(`stageContent-${i}`);
      if (el) {
        if (i === state.stage) {
          el.classList.remove('hidden');
        } else {
          el.classList.add('hidden');
        }
      }
    }

    // Update Inspector Drawer
    updateInspectorDrawer(state.stage);

    // Render visualizer
    renderStage(state.stage);
  }

  function renderStage(stage) {
    switch (stage) {
      case 1:
        renderStage1Tokenization();
        break;
      case 2:
        renderStage2Embeddings();
        break;
      case 3:
        renderStage3Attention();
        break;
      case 4:
        renderStage4FFN();
        break;
      case 5:
        renderStage5Normalization();
        break;
      case 6:
        renderStage6LMHead();
        break;
      case 7:
        renderStage7Sampling();
        break;
    }
  }

  // ==========================================================================
  // Stage 1: Tokenization Renderers
  // ==========================================================================
  function renderStage1Tokenization() {
    tokenChipsDisplay.innerHTML = '';
    tokenTableBody.innerHTML = '';

    if (state.tokens.length === 0) {
      tokenChipsDisplay.innerHTML = '<span class="hint-text">Type a prompt above to see tokenization.</span>';
      return;
    }

    state.tokens.forEach((t, idx) => {
      // 1. Chip
      const chip = document.createElement('div');
      chip.className = `token-chip hue-${t.hueIndex}`;
      chip.innerHTML = `
        <span class="token-text">${escapeHtml(t.text)}</span>
        <span class="token-id-tag">ID: ${t.id}</span>
      `;
      chip.title = `Token index ${idx} | ${t.byteLength} bytes`;
      tokenChipsDisplay.appendChild(chip);

      // 2. Table Row
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>#${idx}</td>
        <td><code>${escapeHtml(t.text)}</code></td>
        <td>${t.id}</td>
        <td>${t.byteLength} B</td>
        <td><span class="badge-small">${t.type}</span></td>
      `;
      tokenTableBody.appendChild(tr);
    });
  }

  // ==========================================================================
  // Stage 2: Embeddings & RoPE Heatmap
  // ==========================================================================
  function renderStage2Embeddings() {
    embeddingHeatmap.innerHTML = '';
    const matrix = engine.computeEmbeddings(state.tokens);

    const grid = document.createElement('div');
    grid.className = 'matrix-grid';

    matrix.forEach((row, rowIdx) => {
      const rowDiv = document.createElement('div');
      rowDiv.className = 'matrix-row';

      const label = document.createElement('div');
      label.className = 'matrix-row-label';
      label.textContent = state.tokens[rowIdx]?.text || `T${rowIdx}`;
      label.title = `Token ${rowIdx}: "${state.tokens[rowIdx]?.text}"`;
      rowDiv.appendChild(label);

      for (let c = 0; c < row.length; c++) {
        const val = row[c];
        const cell = document.createElement('div');
        cell.className = 'matrix-cell';
        
        // Ethereal Blue Gradient coloring based on value (-1 to +1)
        const norm = (val + 1) / 2; // 0 to 1
        cell.style.backgroundColor = getEtherealColor(norm);

        cell.addEventListener('mouseenter', () => {
          embeddingCellInspector.textContent = `Token "${state.tokens[rowIdx]?.text}" [dim ${c}]: ${val.toFixed(4)}`;
        });

        rowDiv.appendChild(cell);
      }
      grid.appendChild(rowDiv);
    });

    embeddingHeatmap.appendChild(grid);

    // Draw RoPE Phase Canvas
    drawRoPECanvas();
  }

  function getEtherealColor(t) {
    // 0 = Dark Navy (#03045e), 0.5 = Sea Blue (#00b4d8), 1.0 = Ice White (#ffffff)
    if (t < 0.5) {
      const factor = t * 2;
      return interpolateColor('#03045e', '#00b4d8', factor);
    } else {
      const factor = (t - 0.5) * 2;
      return interpolateColor('#00b4d8', '#ffffff', factor);
    }
  }

  function interpolateColor(color1, color2, factor) {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);
    const r = Math.round(c1.r + factor * (c2.r - c1.r));
    const g = Math.round(c1.g + factor * (c2.g - c1.g));
    const b = Math.round(c1.b + factor * (c2.b - c1.b));
    return `rgb(${r}, ${g}, ${b})`;
  }

  function hexToRgb(hex) {
    let clean = hex.replace('#', '');
    if (clean.length === 3) clean = clean.split('').map(c => c + c).join('');
    const num = parseInt(clean, 16);
    return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
  }

  function drawRoPECanvas() {
    if (!ropeCanvas) return;
    const ctx = ropeCanvas.getContext('2d');
    const w = ropeCanvas.width;
    const h = ropeCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = 65;

    // Draw coordinate circle
    ctx.strokeStyle = 'rgba(144, 224, 239, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Axes
    ctx.beginPath();
    ctx.moveTo(cx - radius - 10, cy);
    ctx.lineTo(cx + radius + 10, cy);
    ctx.moveTo(cx, cy - radius - 10);
    ctx.lineTo(cx, cy + radius + 10);
    ctx.stroke();

    // Draw rotating token vectors
    const N = state.tokens.length;
    for (let i = 0; i < N; i++) {
      // Rotation angle proportional to token position
      const angle = (i * 0.45) % (Math.PI * 2);
      const vx = cx + Math.cos(angle) * (radius * 0.85);
      const vy = cy - Math.sin(angle) * (radius * 0.85);

      // Vector line
      ctx.strokeStyle = i === N - 1 ? '#caf0f8' : 'rgba(0, 180, 216, 0.65)';
      ctx.lineWidth = i === N - 1 ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(vx, vy);
      ctx.stroke();

      // Point
      ctx.fillStyle = i === N - 1 ? '#ffffff' : '#48cae4';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#00b4d8';
      ctx.beginPath();
      ctx.arc(vx, vy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.fillStyle = '#caf0f8';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(`m=${i}`, vx + 6, vy + 4);
    }
  }

  // ==========================================================================
  // Stage 3: Multi-Head Attention Renderers
  // ==========================================================================
  function renderStage3Attention() {
    attnMatrixContainer.innerHTML = '';
    attnTokenLinks.innerHTML = '';

    const attnResult = engine.computeAttention(state.tokens, state.activeHead);
    const matrix = attnResult.matrix;
    const N = state.tokens.length;

    if (N === 0) return;

    // 1. Matrix Table
    const table = document.createElement('table');
    table.className = 'attn-grid-table';

    // Header row
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>Q \\ K</th>';
    state.tokens.forEach(t => {
      const th = document.createElement('th');
      th.textContent = t.text.trim();
      th.title = t.text;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body rows
    const tbody = document.createElement('tbody');
    matrix.forEach((row, i) => {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.textContent = state.tokens[i].text.trim();
      tr.appendChild(th);

      for (let j = 0; j < N; j++) {
        const td = document.createElement('td');
        td.className = 'attn-cell';

        if (j > i) {
          td.classList.add('masked');
          td.textContent = '—';
          td.title = `Masked future token (causal)`;
        } else {
          const weight = row[j];
          td.textContent = (weight * 100).toFixed(0) + '%';
          td.style.backgroundColor = `rgba(0, 180, 216, ${Math.max(0.12, weight * 0.95)})`;
          td.style.color = weight > 0.35 ? '#ffffff' : '#caf0f8';

          td.addEventListener('mouseenter', () => {
            attnCellInfo.innerHTML = `Query <strong>"${state.tokens[i].text.trim()}"</strong> attends to Key <strong>"${state.tokens[j].text.trim()}"</strong>: <strong>${(weight * 100).toFixed(2)}%</strong> (raw score = ${attnResult.rawScores[i][j].toFixed(3)})`;
            updateTokenAttentionBars(row, i);
          });
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    attnMatrixContainer.appendChild(table);

    // Initial bars for last token
    updateTokenAttentionBars(matrix[N - 1], N - 1);
  }

  function updateTokenAttentionBars(weightsRow, queryIdx) {
    attnTokenLinks.innerHTML = '';
    const qToken = state.tokens[queryIdx];

    state.tokens.forEach((t, j) => {
      if (j > queryIdx) return; // masked
      const w = weightsRow[j];
      const item = document.createElement('div');
      item.className = 'attn-token-bar-item';
      item.innerHTML = `
        <div class="token-bar-meta">
          <span>"${t.text.trim()}"</span>
          <span>${(w * 100).toFixed(1)}%</span>
        </div>
        <div class="token-bar-track">
          <div class="token-bar-fill" style="width: ${w * 100}%"></div>
        </div>
      `;
      attnTokenLinks.appendChild(item);
    });
  }

  // Head explanations
  const headExplanations = [
    "Head 1: Local Recency - Specializes in strong attention to immediate predecessor tokens.",
    "Head 2: Syntactic & Modifier Linking - Connects adjectives and descriptive tokens with their target nouns.",
    "Head 3: Broad Global Context - Anchors context by attending to beginning / delimiter tokens.",
    "Head 4: Pattern Induction - Detects recurring sequences and syntactic rhyme structures."
  ];

  headBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      headBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeHead = parseInt(btn.getAttribute('data-head'));
      headExplanationText.textContent = headExplanations[state.activeHead];
      renderStage3Attention();
    });
  });

  // ==========================================================================
  // Stage 4: FFN & Activation Canvas
  // ==========================================================================
  function renderStage4FFN() {
    drawActivationCanvas();
  }

  function drawActivationCanvas() {
    if (!activationCanvas) return;
    const ctx = activationCanvas.getContext('2d');
    const w = activationCanvas.width;
    const h = activationCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2 + 20;
    const scaleX = 40;
    const scaleY = 32;

    // Grid axes
    ctx.strokeStyle = 'rgba(144, 224, 239, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    // Plot curve
    ctx.strokeStyle = '#00b4d8';
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#00b4d8';
    ctx.beginPath();

    for (let px = 0; px < w; px++) {
      const x = (px - cx) / scaleX;
      const y = engine.evalActivation(x, engine.activation);
      const py = cy - y * scaleY;

      if (px === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Axis labels
    ctx.fillStyle = 'rgba(202, 240, 248, 0.6)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('x = 0', cx + 4, cy + 14);
    ctx.fillText('f(x)', cx + 4, 14);
  }

  activationCanvas.addEventListener('mousemove', (e) => {
    const rect = activationCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const cx = activationCanvas.width / 2;
    const scaleX = 40;
    const x = ((px * (activationCanvas.width / rect.width)) - cx) / scaleX;
    const y = engine.evalActivation(x, engine.activation);
    plotHoverVal.textContent = `Input x: ${x.toFixed(2)} ➔ Output f(x): ${y.toFixed(3)} (${engine.activation.toUpperCase()})`;
  });

  btnActSwiGLU.addEventListener('click', () => {
    setActivation('swiglu', btnActSwiGLU);
  });
  btnActGELU.addEventListener('click', () => {
    setActivation('gelu', btnActGELU);
  });
  btnActReLU.addEventListener('click', () => {
    setActivation('relu', btnActReLU);
  });

  function setActivation(act, btn) {
    [btnActSwiGLU, btnActGELU, btnActReLU].forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    engine.activation = act;
    drawActivationCanvas();
  }

  // ==========================================================================
  // Stage 5: Normalization & Residual Highway
  // ==========================================================================
  function renderStage5Normalization() {
    // Synchronize UI values
    gammaVal.textContent = engine.gamma.toFixed(1);
    if (engine.normType === 'rmsnorm') {
      cardRMSNorm.classList.add('active-card');
      cardLayerNorm.classList.remove('active-card');
      btnNormRMS.classList.add('active');
      btnNormLayer.classList.remove('active');
    } else {
      cardRMSNorm.classList.remove('active-card');
      cardLayerNorm.classList.add('active-card');
      btnNormRMS.classList.remove('active');
      btnNormLayer.classList.add('active');
    }
  }

  btnNormRMS.addEventListener('click', () => {
    engine.normType = 'rmsnorm';
    renderStage5Normalization();
  });
  btnNormLayer.addEventListener('click', () => {
    engine.normType = 'layernorm';
    renderStage5Normalization();
  });

  gammaSlider.addEventListener('input', (e) => {
    engine.gamma = parseFloat(e.target.value);
    renderStage5Normalization();
  });

  // ==========================================================================
  // Stage 6: LM Head Unembedding Logits
  // ==========================================================================
  function renderStage6LMHead() {
    lastTokenVectorStrip.innerHTML = '';
    logitsBarsList.innerHTML = '';

    // 1. Draw hidden state vector cells (64 dimensions)
    const hidden = engine.computeEmbeddings(state.tokens);
    const lastVec = hidden[hidden.length - 1] || new Float32Array(engine.dModel);

    for (let i = 0; i < lastVec.length; i++) {
      const cell = document.createElement('div');
      cell.className = 'vector-strip-cell';
      const norm = (lastVec[i] + 1) / 2;
      cell.style.backgroundColor = getEtherealColor(norm);
      cell.title = `h_last[${i}] = ${lastVec[i].toFixed(4)}`;
      lastTokenVectorStrip.appendChild(cell);
    }

    // 2. Candidate unscaled logits
    const candidates = engine.computeCandidateLogits(state.prompt);
    candidates.forEach(c => {
      const row = document.createElement('div');
      row.className = 'logit-bar-row';
      const pctWidth = Math.min(100, Math.max(10, c.logit * 13));
      row.innerHTML = `
        <div class="logit-bar-meta">
          <span class="logit-token-name">"${escapeHtml(c.text)}"</span>
          <span class="logit-score-val">+${c.logit.toFixed(2)}</span>
        </div>
        <div class="logit-bar-track">
          <div class="logit-bar-fill" style="width: ${pctWidth}%"></div>
        </div>
      `;
      logitsBarsList.appendChild(row);
    });
  }

  // ==========================================================================
  // Stage 7: Sampling & Generation
  // ==========================================================================
  function renderStage7Sampling() {
    probDistributionList.innerHTML = '';

    const candidates = engine.computeCandidateLogits(state.prompt);
    const samplingResult = engine.sampleFromCandidates(
      candidates,
      engine.temperature,
      engine.topK,
      engine.topP
    );

    // Update filter badge
    activeFilterBadge.textContent = `Top-k (${engine.topK}) & Top-p (${engine.topP.toFixed(2)}) Active`;

    // Render candidate probability bars
    samplingResult.allCandidates.forEach(c => {
      const row = document.createElement('div');
      row.className = 'prob-bar-row';
      const isEligible = c.passedTopK && c.passedTopP;
      if (!isEligible) row.classList.add('filtered-out');

      const pct = (c.prob * 100).toFixed(1);
      row.innerHTML = `
        <div class="prob-bar-meta">
          <span class="prob-token-name">"${escapeHtml(c.text)}"</span>
          <span class="prob-pct-val">${pct}% ${!isEligible ? '(pruned)' : ''}</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct}%"></div>
        </div>
      `;
      probDistributionList.appendChild(row);
    });

    // Update preview of selected token
    if (samplingResult.selected) {
      selectedTokenDisplay.textContent = `"${samplingResult.selected.text}"`;
      selectedTokenProb.textContent = `(${(samplingResult.selected.prob * 100).toFixed(1)}%)`;
    }
  }

  // Sample and append action
  function sampleAndAppend() {
    const candidates = engine.computeCandidateLogits(state.prompt);
    const result = engine.sampleFromCandidates(
      candidates,
      engine.temperature,
      engine.topK,
      engine.topP
    );

    if (result.selected) {
      const newToken = result.selected.text;
      promptInput.value = promptInput.value + newToken;
      
      // Butterfly spark burst at button position
      const btnRect = btnSampleNext.getBoundingClientRect();
      if (sim) {
        sim.createBurst(btnRect.left + btnRect.width / 2, btnRect.top + btnRect.height / 2, 16);
      }

      runPipeline();
    }
  }

  btnSampleNext.addEventListener('click', sampleAndAppend);
  btnQuickSample.addEventListener('click', sampleAndAppend);

  // Autoregressive Streaming loop
  btnAutoPlay.addEventListener('click', () => {
    state.isPlaying = !state.isPlaying;
    if (state.isPlaying) {
      btnAutoPlay.classList.add('active');
      autoPlayLabel.textContent = "Pause Stream";
      state.playInterval = setInterval(() => {
        if (state.tokens.length > 28) {
          stopAutoPlay();
          return;
        }
        sampleAndAppend();
      }, 900);
    } else {
      stopAutoPlay();
    }
  });

  function stopAutoPlay() {
    state.isPlaying = false;
    clearInterval(state.playInterval);
    btnAutoPlay.classList.remove('active');
    autoPlayLabel.textContent = "Autoregressive Stream";
  }

  // Sliders
  tempSlider.addEventListener('input', (e) => {
    engine.temperature = parseFloat(e.target.value);
    tempVal.textContent = engine.temperature.toFixed(2);
    renderStage7Sampling();
  });

  topKSlider.addEventListener('input', (e) => {
    engine.topK = parseInt(e.target.value);
    topKVal.textContent = engine.topK;
    renderStage7Sampling();
  });

  topPSlider.addEventListener('input', (e) => {
    engine.topP = parseFloat(e.target.value);
    topPVal.textContent = engine.topP.toFixed(2);
    renderStage7Sampling();
  });

  // ==========================================================================
  // Bottom Sequence Bar
  // ==========================================================================
  function updateSequenceDisplay() {
    genSequenceDisplay.innerHTML = '';
    state.tokens.forEach((t, i) => {
      const chip = document.createElement('span');
      chip.className = `token-chip-seq ${i > 7 ? 'generated' : 'base'}`;
      chip.textContent = t.text;
      genSequenceDisplay.appendChild(chip);
    });
    // Auto scroll to right
    genSequenceDisplay.scrollLeft = genSequenceDisplay.scrollWidth;
  }

  btnQuickReset.addEventListener('click', () => {
    promptInput.value = presetSelect.value === 'butterfly' 
      ? "The ethereal blue butterfly dances over the"
      : presetSelect.options[presetSelect.selectedIndex].text;
    stopAutoPlay();
    runPipeline();
  });

  btnResetPrompt.addEventListener('click', () => {
    promptInput.value = "The ethereal blue butterfly dances over the";
    stopAutoPlay();
    runPipeline();
  });

  // Preset Selector
  presetSelect.addEventListener('change', () => {
    const val = presetSelect.value;
    const presets = {
      butterfly: "The ethereal blue butterfly dances over the",
      reasoning: "To discover the truth we must explore",
      code: "def generate_starlight(prompt): return",
      simple: "The quick azure fox jumps over"
    };
    promptInput.value = presets[val] || presets.butterfly;
    stopAutoPlay();
    runPipeline();
  });

  // Model Selector
  modelSelect.addEventListener('change', () => {
    engine.architecture = modelSelect.value;
    if (modelSelect.value === 'llama3') {
      engine.posScheme = 'rope';
      engine.activation = 'swiglu';
      engine.normType = 'rmsnorm';
    } else if (modelSelect.value === 'gpt2') {
      engine.posScheme = 'learned';
      engine.activation = 'gelu';
      engine.normType = 'layernorm';
    } else {
      engine.posScheme = 'rope';
      engine.activation = 'swiglu';
      engine.normType = 'rmsnorm';
    }
    runPipeline();
  });

  // Positional Scheme Buttons
  btnPosRoPE.addEventListener('click', () => {
    [btnPosRoPE, btnPosSinusoidal, btnPosLearned].forEach(b => b.classList.remove('active'));
    btnPosRoPE.classList.add('active');
    engine.posScheme = 'rope';
    renderStage2Embeddings();
  });

  btnPosSinusoidal.addEventListener('click', () => {
    [btnPosRoPE, btnPosSinusoidal, btnPosLearned].forEach(b => b.classList.remove('active'));
    btnPosSinusoidal.classList.add('active');
    engine.posScheme = 'sinusoidal';
    renderStage2Embeddings();
  });

  btnPosLearned.addEventListener('click', () => {
    [btnPosRoPE, btnPosSinusoidal, btnPosLearned].forEach(b => b.classList.remove('active'));
    btnPosLearned.classList.add('active');
    engine.posScheme = 'learned';
    renderStage2Embeddings();
  });

  // ==========================================================================
  // Stepper Bar Clicks & Inspector Drawer
  // ==========================================================
  stagesTrack.querySelectorAll('.stage-node').forEach(node => {
    node.addEventListener('click', () => {
      const s = parseInt(node.getAttribute('data-stage'));
      setStage(s);
    });
  });

  btnPrevStage.addEventListener('click', () => setStage(state.stage - 1));
  btnNextStage.addEventListener('click', () => setStage(state.stage + 1));

  function updateInspectorDrawer(stageNum) {
    const data = stageInspectorData[stageNum];
    if (!data) return;

    inspectorMathFormula.textContent = data.formula;
    
    // Tensors
    inspectorTensorShapes.innerHTML = '';
    data.tensors.forEach(t => {
      const item = document.createElement('div');
      item.className = 'tensor-item';
      item.innerHTML = `
        <span class="tensor-name">${t.name}</span>
        <span class="tensor-dims">${t.dims}</span>
      `;
      inspectorTensorShapes.appendChild(item);
    });

    inspectorIntuition.textContent = data.intuition;

    inspectorArchitectureNotes.innerHTML = '';
    data.architectures.forEach(note => {
      const li = document.createElement('li');
      li.textContent = note;
      inspectorArchitectureNotes.appendChild(li);
    });
  }

  // Inspector Toggle
  function toggleInspector() {
    state.inspectorOpen = !state.inspectorOpen;
    if (state.inspectorOpen) {
      deepInspectorAside.style.display = 'flex';
      mainLayout.classList.remove('collapsed-inspector');
      deepInspectorBtn.classList.add('active');
    } else {
      deepInspectorAside.style.display = 'none';
      mainLayout.classList.add('collapsed-inspector');
      deepInspectorBtn.classList.remove('active');
    }
  }

  deepInspectorBtn.addEventListener('click', toggleInspector);
  btnCloseInspector.addEventListener('click', toggleInspector);

  // Butterfly mode toggle
  butterflyModeToggle.addEventListener('click', () => {
    if (sim) {
      const isActive = sim.toggle();
      if (isActive) {
        butterflyModeToggle.classList.add('active');
        butterflyModeToggle.querySelector('.btn-label').textContent = 'Butterflies: On';
      } else {
        butterflyModeToggle.classList.remove('active');
        butterflyModeToggle.querySelector('.btn-label').textContent = 'Butterflies: Off';
      }
    }
  });

  // Prompt input listener
  promptInput.addEventListener('input', () => {
    runPipeline();
  });

  // Helper
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Initialize
  runPipeline();
  setStage(1);
});

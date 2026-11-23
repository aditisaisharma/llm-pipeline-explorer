# 🦋 MorphoLLM: Ethereal LLM Pipeline Explorer

> A personalized, ethereal interactive visualizer for the Large Language Model transformer inference pipeline, decorated with fluttering morpho butterflies, stardust trails, and a sea blue, navy, and crystal ice palette.

---

## ✨ Features & Architecture

### 1. 🔤 Subword Tokenization (BPE)
- Real-time token breakdown with dynamic Byte-Pair Encoding logic.
- Visual token chips color-coded by byte semantics and token IDs.
- Byte length, UTF-8 prefix inspection, and vocabulary lookup statistics.

### 2. 🧬 Embeddings & Rotary Position Encoding (RoPE)
- $X_0 \in \mathbb{R}^{seq \times d_{model}}$ vector heatmap with cell hover inspector.
- **RoPE (Rotary Position Embedding)**: Interactive 2D phase-space canvas showing token vector coordinate rotation ($m \cdot \theta_i$).
- Positional scheme switcher: **RoPE** (Llama 3, Mistral) vs. **Sinusoidal** (Transformer original) vs. **Learned Absolute** (GPT-2).

### 3. 👁️ Multi-Head Causal Self-Attention
- Full attention projections ($Q, K, V$).
- Scaled Dot-Product matrix calculation: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right)$.
- Lower-triangular causal mask preventing future token peeking.
- Interactive $N \times N$ attention weight matrix: hover over any cell $(i, j)$ to see query-key attention distribution.
- **Multi-Head Selector**: Toggle between 4 heads with distinct roles (Local Recency, Syntactic/Modifier Linking, Global Anchoring, Pattern Induction).

### 4. ⚡ Feed-Forward Network (FFN / MLP) & SwiGLU
- SwiGLU architecture schematic with gate and up-projections ($d_{ffn} = \frac{8}{3} d_{model}$).
- Interactive activation curve plotter: toggle between **SwiGLU (SiLU)**, **GELU** (GPT-2/4), and **ReLU**.

### 5. 🌊 RMSNorm & The Residual Stream
- Visual comparison of **RMSNorm** (Llama 3/Mistral/DeepSeek) vs. **LayerNorm** (GPT-2).
- Live scaling parameter ($\gamma$) slider.
- Residual highway flow animation demonstrating the unimpeded gradient flow.

### 6. 📊 LM Head Unembedding & Logits
- Last-token hidden representation ($h_{last}$) vector cell viewer.
- Raw unscaled logit projection bar chart across vocabulary candidates.

### 7. 🎲 Softmax & Autoregressive Sampling
- Dynamic temperature slider ($T \in [0.05, 2.0]$) shaping distribution flatness.
- Top-$k$ and Nucleus Top-$p$ cutoff controls with active pruning markers.
- **"Sample & Append Next Token"** button with glowing butterfly burst effect.
- **"Autoregressive Stream"** autoplay loop that iteratively predicts tokens and updates all pipeline stages.

---

## 🎨 Color Palette & Aesthetic Design

- **Deepest Navy Night**: `#030712`, `#060e24`, `#0a1636`
- **Sea Blue & Aquamarine**: `#005f73`, `#0a9396`, `#0077b6`, `#00b4d8`, `#48cae4`
- **Light Blue & Ethereal Ice**: `#90e0ef`, `#caf0f8`, `#e0f2fe`, `#f0f9ff`
- **Iridescent Butterfly Accents**: Soft lavender-blue (`#c7d2fe`) and glowing stardust particles (`#caf0f8`).
- **Glassmorphism**: Translucent frosted cards (`backdrop-filter: blur(20px)`), subtle glowing borders, and floating interactive butterflies on HTML5 Canvas.

---

## 🚀 Running the Website

### Option 1: Direct File Opening
Double-click `index.html` in your file manager or open it in any web browser (Chrome, Safari, Firefox, Edge). It is 100% self-contained with no build steps or dependencies.

### Option 2: Python Local Server
```bash
python3 serve.py
# or
python3 -m http.server 8080
```
Then visit `http://localhost:8080` in your browser.

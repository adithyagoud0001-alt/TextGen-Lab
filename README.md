# TextGen Lab: Interactive Transformer-Based Text Generation System 🧠

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-DistilGPT--2-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/distilgpt2)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An educational Generative AI laboratory and interactive web application designed to demonstrate the inner mathematical and architectural workings of **Transformer-based language models**. Built with **PyTorch**, **Hugging Face Transformers (`DistilGPT-2`)**, and **Streamlit**, this project runs **100% locally on CPU/GPU without requiring external API keys or cloud dependencies**.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [Core Modules & Features](#5-core-modules--features)
6. [Mathematical Foundations](#6-mathematical-foundations)
7. [Installation & Setup](#7-installation--setup)
8. [Execution Guide](#8-execution-guide)
9. [Sample Prompts for Demonstration](#9-sample-prompts-for-demonstration)
10. [Student Viva Voce & Defense Script](#10-student-viva-voce--defense-script)
11. [Limitations & Future Enhancements](#11-limitations--future-enhancements)
12. [Project Structure](#12-project-structure)

---

## 1. Project Overview

Modern Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini have revolutionized technology, but their sheer scale often turns them into opaque "black boxes" for engineering students and researchers.

**TextGen Lab** peels back the abstraction layer. Using **DistilGPT-2** (an 82-million parameter causal language model), TextGen Lab provides an interactive, visual playground where users can:
- Generate contextually coherent text from arbitrary input prompts.
- Control sampling dynamics in real-time (Temperature, Top-K, Repetition Penalty).
- Inspect the exact raw model logits and their transformation into a valid probability distribution using **Softmax**.
- Explore the end-to-end Transformer pipeline: from Byte-Pair Encoding (BPE) tokenization to multi-head self-attention and autoregressive decoding.
- Conduct side-by-side experiments comparing low, balanced, and high temperature sampling regimes.

---

## 2. Problem Statement & Objectives

### The Problem
Traditional AI coursework often treats generative language models either through purely abstract mathematics (equations without visual grounding) or through commercial API wrappers (where hidden state matrices, logits, and token distributions cannot be inspected or altered).

### Project Objectives
1. **Demystify Text Generation:** Show how an autoregressive Transformer predicts text one token at a time based on statistical language patterns.
2. **Visualize Logits to Probabilities:** Demonstrate the exact Softmax equation and verify that probabilities across 50,257 vocabulary tokens sum strictly to $1.0$.
3. **Analyze Sampling Dynamics:** Interactively prove how Temperature sharpens or flattens probability curves, and how Top-K filtering prevents low-probability tail degeneration.
4. **Zero-API Dependency:** Ensure the application runs completely offline on standard consumer laptop CPUs without requiring paid API tokens or NVIDIA CUDA GPUs.

---

## 3. Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10 / 3.11 | Standard language for AI/ML and deep learning research. |
| **Deep Learning Framework** | PyTorch (`torch`) | Tensor computations, automatic differentiation, and neural network primitives. |
| **Pretrained Model** | `DistilGPT-2` | 82M parameter distilled GPT-2. Fits into standard RAM (~350MB), runs fast on CPU, retains >95% GPT-2 language capability. |
| **Model Library** | Hugging Face `transformers` | Industry-standard library for loading pretrained tokenizers and autoregressive language heads. |
| **Web Frontend** | Streamlit | Rapid, modern reactive web UI with seamless Python state management and caching. |
| **Data Visualization** | Plotly & Matplotlib | Interactive HTML5 charts with hover metadata, percentage labels, and high-DPI architecture diagrams. |
| **Numerical Computing** | NumPy | Numerically stable vectorized mathematical calculations for Softmax, shifts, and entropy. |

---

## 4. System Architecture

The entire text-generation pipeline follows the **Decoder-Only Autoregressive Transformer** design pattern:

```
+-----------------------------------------------------------------------------------+
|                           INPUT PROMPT ("Artificial intelligence will")           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 1. BYTE-PAIR ENCODING (BPE) TOKENIZER                                             |
|    - Maps text into token IDs: [7928, 4430, 481]                                  |
|    - Vocabulary Size: 50,257 subwords                                             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. EMBEDDING LAYER                                                                |
|    - Token Embeddings (768-d vector lookup)                                       |
|    - Learned Positional Embeddings (0 to 1023 sequence positions)                 |
|    - Output: X_0 = Token_Embed + Pos_Embed                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. TRANSFORMER BLOCKS (x6 Layers in DistilGPT-2)                                  |
|    - LayerNorm -> Multi-Head Masked Causal Self-Attention (12 heads, d_k=64)      |
|    - Residual Skip Connection (X + Attention(X))                                  |
|    - LayerNorm -> Feed-Forward Network (768 -> 3072 -> 768 with GELU)             |
|    - Residual Skip Connection (X + FFN(X))                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. LANGUAGE MODELING (LM) PROJECTION HEAD                                         |
|    - Final LayerNorm                                                              |
|    - Linear Projection: R^768 -> R^50,257                                         |
|    - Output: Raw Unnormalized Logits (z_1, z_2, ..., z_50257)                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. TEMPERATURE SCALING & SOFTMAX NORMALIZATION                                    |
|    - Scaled Logits: z_i' = z_i / T                                                |
|    - Softmax: P(w_i) = exp(z_i') / Sum_j(exp(z_j'))                              |
|    - Top-K Filtering & Repetition Penalty Masking                                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 6. NEXT-TOKEN SELECTION & AUTOREGRESSIVE FEEDBACK LOOP                            |
|    - Greedy: w* = argmax(P)   OR   Sampling: w ~ P                                |
|    - Selected Token: " shape"                                                     |
|    - Context = Context + " shape"  -->  Repeat until EOS or max_tokens reached    |
+-----------------------------------------------------------------------------------+
```

An automatically generated high-resolution diagram is saved at `assets/architecture.png`.

---

## 5. Core Modules & Features

### Module 1: Interactive Text Generation (`app.py` & `generation.py`)
- User-friendly prompt input with instant one-click starter prompts.
- Visual distinction between user prompt (gray badge) and newly generated tokens (blue highlight).
- Real-time generation telemetry: Prompt tokens, generated tokens, latency (seconds), and throughput (tokens/second).
- Caching with `@st.cache_resource` ensures DistilGPT-2 loads into RAM exactly once.

### Module 2: Generation Parameter Controls
Interactive sidebar controls for fine-tuning text generation:
- **Temperature ($0.0 \le T \le 2.0$):** $T=0.0$ triggers deterministic Greedy Decoding.
- **Top-K Sampling ($0 \le K \le 100$):** Limits candidate choices to the top $K$ tokens.
- **Repetition Penalty ($1.0 \le \rho \le 2.0$):** Penalizes already generated tokens to prevent repeating loops.
- **Max New Tokens ($5 \le N \le 120$):** Restricts generation length.
- **Random Seed:** Allows exact reproducibility of stochastic sampling.

### Module 3: Next-Token Probability Visualization (`visualization.py`)
- Extracts raw model logits for the final sequence position via forward pass.
- Computes full-vocabulary Softmax distribution.
- Renders an interactive Plotly horizontal bar chart showing the Top 5 or Top 10 predicted tokens with percentage labels.
- The highest probability candidate (#1 rank) is highlighted in glowing emerald green.
- Shows Shannon Entropy (distribution uncertainty) and proves the probability sum equals $1.0000$.

### Module 4: Transformer Architecture Explorer
- Comprehensive breakdown of all 6 architectural stages: Tokenization, Embeddings, Multi-Head Causal Attention, FFN, Projection Head, and Autoregression.
- Visual pipeline diagram and hyperparameter specifications of DistilGPT-2.

### Module 5: Temperature Experimentation Lab
- Runs concurrent generation across three temperature settings:
  1. **Low ($T=0.3$):** Conservative, deterministic, low vocabulary diversity.
  2. **Medium ($T=0.8$):** Natural linguistic balance of coherence and creativity.
  3. **High ($T=1.3$):** Flattened probability curve, exploratory, higher entropy.
- Side-by-side metric comparison: Lexical diversity (unique word ratio), latency, and generated text.
- Grouped bar chart proving how the exact same candidate tokens have their probabilities altered by $T$.

---

## 6. Mathematical Foundations

### A. The Softmax Function
The Softmax function maps an unconstrained $K$-dimensional vector of real-valued logits $z = [z_1, z_2, \dots, z_K] \in \mathbb{R}^K$ into a valid probability distribution $P \in (0, 1)^K$:

$$\text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}$$

#### Mathematical Properties:
1. **Positivity:** $\exp(x) > 0$ for all real $x$, ensuring probabilities are strictly non-negative.
2. **Partition Normalization:** The denominator $\sum_{j=1}^K \exp(z_j)$ acts as a normalizing constant (partition function), ensuring:
   $$\sum_{i=1}^{K} P_i = 1.0$$
3. **Monotonicity:** If $z_a > z_b$, then $\text{Softmax}(z_a) > \text{Softmax}(z_b)$.

#### Numerical Stability Implementation:
To prevent floating-point overflow when exponentiating large positive logits (e.g., $\exp(500) \to \infty$), we subtract the maximum logit before computing exponentials:

$$z_i' = z_i - \max_{j} z_j$$
$$\text{Softmax}(z_i) = \frac{\exp(z_i')}{\sum_{j=1}^K \exp(z_j')}$$

### B. Scaled Dot-Product Attention
Attention allows the model to dynamically weight the importance of different words in the context window:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

Where:
- $Q = X W_Q \in \mathbb{R}^{N \times d_k}$ (Query Matrix)
- $K = X W_K \in \mathbb{R}^{N \times d_k}$ (Key Matrix)
- $V = X W_V \in \mathbb{R}^{N \times d_v}$ (Value Matrix)
- $d_k = 64$ (dimension of each attention head)

#### Why Scale by $\sqrt{d_k}$?
For large head dimensions $d_k$, the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has a variance equal to $d_k$ (assuming components are independent random variables with zero mean and unit variance). Without scaling by $\frac{1}{\sqrt{d_k}}$, large values of $Q K^T$ push the Softmax function into regions with near-zero gradients, leading to severe vanishing gradients.

### C. Temperature Scaling
Temperature $T > 0$ modifies the logits prior to Softmax:

$$P(w_i \mid \text{context}) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

- **As $T \to 0$:** The largest logit dominates completely, approaching the indicator function (one-hot argmax, Greedy Search).
- **When $T = 1.0$:** Standard Softmax distribution learned during training.
- **As $T \to \infty$:** All exponentiated values approach $1.0$, resulting in a completely uniform distribution $P_i = \frac{1}{V}$, maximizing entropy.

### D. Top-K Sampling
Given sorted probabilities $P_{(1)} \ge P_{(2)} \ge \dots \ge P_{(V)}$:
1. Truncate distribution to $S_K = \{w_{(1)}, w_{(2)}, \dots, w_{(K)}\}$.
2. Renormalize probabilities over $S_K$:
   $$P'(w_i) = \frac{P(w_i)}{\sum_{w_j \in S_K} P(w_j)}$$
3. Sample from $P'$.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10 or 3.11 installed.
- Git (optional, for cloning).

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "c:\Users\G.Adithya Goud\Desktop\AI-Project"
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
For CPU systems (recommended for laptop demonstration):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For NVIDIA GPU systems (optional):
```bash
pip install torch
pip install -r requirements.txt
```

---

## 8. Execution Guide

### Launching the Web Application
Run the Streamlit application using:
```bash
streamlit run app.py
```

The application will launch in your default web browser at:
```
http://localhost:8501
```

### Running the Automated Test Suite
To verify that all modules, model loading, Softmax normalization, and diagram generation work correctly:
```bash
python test_pipeline.py
```

Expected output:
```
======================================================================
TextGen Lab - Verification Test Suite
======================================================================
  ✓ All required libraries and local modules imported successfully.
  ✓ Active Computation Device: cpu
  ✓ Model Loaded: distilgpt2
  ✓ Parameters: 81.91M
  ✓ Vocabulary Size: 50257
  ✓ Empty prompt handled gracefully.
  ✓ Greedy Output verified.
  ✓ Softmax Axiom Verified: sum(P) == 1.0000
  ✓ Temperature experiment verified.
  ✓ Plotly charts created without error.
  ✓ Architecture diagram saved at assets/architecture.png
======================================================================
🎉 ALL 9 TEST SUITES PASSED PERFECTLY!
======================================================================
```

---

## 9. Sample Prompts for Demonstration

Here are five tested prompts that highlight the model's linguistic and reasoning capabilities:

1. **Scientific Reasoning:**
   ```
   Prompt: "The fundamental difference between machine learning and traditional programming is"
   Expected Continuation: "...that machine learning systems learn patterns directly from data rather than relying on hand-crafted rules."
   ```

2. **Philosophical Inquiry:**
   ```
   Prompt: "Artificial intelligence will shape the future of human society by"
   Expected Continuation: "...automating repetitive labor and transforming scientific discovery."
   ```

3. **Narrative Storytelling:**
   ```
   Prompt: "Once upon a time in a silent laboratory, the robotic assistant began to"
   Expected Continuation: "...observe the stars through the glass dome above."
   ```

4. **Physics & Astronomy:**
   ```
   Prompt: "To understand the laws of the universe, scientists must"
   Expected Continuation: "...conduct rigorous experiments and observe cosmological phenomena."
   ```

5. **Linguistic Loop Test (Repetition Penalty Demo):**
   ```
   Prompt: "The train was late because the train was late because"
   Setting: Set Repetition Penalty to 1.0 (observe infinite repetition), then raise to 1.3 (observe model breaking the loop).
   ```

---

## 10. Student Viva Voce & Defense Script

When presenting this project to external examiners or professors, use this structured 2-minute walkthrough:

### 2-Minute Presentation Walkthrough:
> **1. Problem & Context (20s):**  
> *"Good morning, sir/madam. Today I am demonstrating **TextGen Lab**, an interactive system that demystifies how Generative AI models generate text using Transformer architectures. Rather than calling black-box cloud APIs, our system executes an 82-million parameter **DistilGPT-2** model locally in PyTorch."*
>
> **2. Live Text Generation (30s):**  
> *(Show Tab 1, enter a sample prompt, click Generate)*  
> *"Here in Tab 1, the model performs autoregressive decoding. The prompt is encoded using Byte-Pair Encoding into token IDs, passed through 6 Transformer layers, and output at over 20 tokens per second on this laptop CPU."*
>
> **3. Softmax & Next-Token Distribution (30s):**  
> *(Show Tab 2, point to the Plotly bar chart)*  
> *"In Tab 2, we expose the mathematical core. The final linear head computes 50,257 raw logits. Applying Softmax transforms them into valid probabilities. Notice our vocabulary sum metric is exactly 1.0000, fulfilling the probability axiom. We highlight the top predicted candidate."*
>
> **4. Temperature Experiment (30s):**  
> *(Show Tab 3, point to the three comparison cards)*  
> *"In Tab 3, we demonstrate the physical meaning of Temperature ($T$). At $T=0.3$, probabilities are sharply peaked, giving deterministic output. At $T=0.8$, we achieve human-like balance. At $T=1.3$, the distribution flattens, increasing entropy and exploratory word choices."*
>
> **5. Conclusion (10s):**  
> *"This project bridges theoretical deep learning mathematics with interactive software engineering."*

### Top 5 Viva Questions & Answers:
1. **Q: Why does DistilGPT-2 use Masked (Causal) Attention instead of Bi-directional Attention like BERT?**  
   *A:* BERT is an *encoder* designed for understanding (masked language modeling where left and right context are visible). GPT-2 is an *autoregressive decoder* designed for sequence generation; a word at position $t$ must never see position $t+1$, so future tokens are masked with $-\infty$.
2. **Q: Why does Softmax require logit shifting $\exp(z_i - \max(z))$?**  
   *A:* Modern 32-bit floats overflow for $\exp(x)$ when $x > 88.7$. Shifting by $\max(z)$ ensures the highest exponent is $\exp(0) = 1$, preventing numerical `NaN` or `inf` errors while producing identical mathematical probabilities.
3. **Q: What is Byte-Pair Encoding (BPE)?**  
   *A:* BPE is a subword tokenization algorithm that iteratively merges the most frequent pairs of characters. It handles out-of-vocabulary words by breaking them into known subwords, keeping the vocabulary compact (50,257 tokens).
4. **Q: How does Top-K sampling differ from Top-P (Nucleus) sampling?**  
   *A:* Top-K selects a fixed number ($K$) of highest-ranked tokens regardless of how flat or steep the distribution is. Top-P dynamically chooses the smallest set of tokens whose cumulative probability exceeds a threshold $P$ (e.g., $0.90$).
5. **Q: What is the computational complexity of Self-Attention?**  
   *A:* $\mathcal{O}(N^2 \cdot d)$, where $N$ is sequence length and $d$ is embedding dimension. Computing $Q K^T$ involves an $N \times N$ matrix multiplication, which is why Transformer context windows have length limits.

---

## 11. Limitations & Future Enhancements

### Current Limitations:
- **Model Scale:** DistilGPT-2 has 82M parameters. It generates grammatically fluent text, but can hallucinate facts and lacks complex logical reasoning compared to 70B+ LLMs.
- **Context Length:** Fixed at 1,024 tokens due to learned positional embeddings.
- **Unidirectional Causal Attention:** Cannot retroactively edit earlier parts of the generated sequence.

### Future Enhancements:
1. **Nucleus (Top-P) & Min-P Sampling:** Add interactive dynamic cutoff sliders alongside Top-K.
2. **Attention Head Visualizer:** Render interactive heatmaps showing the $12 \times 12$ cross-token attention weight matrices.
3. **Beam Search Mode:** Compare single-path sampling against multi-hypothesis beam search tree decoding.
4. **Model Switcher:** Allow one-click switching between `distilgpt2`, `gpt2-medium`, and `TinyLlama-1.1B`.

---

## 12. Project Structure

```
TextGen-Lab/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── model.py                   # Model & tokenizer loading, hardware detection, caching
├── generation.py              # Autoregressive generation, Softmax, next-token probabilities
├── visualization.py           # Plotly interactive charts & Matplotlib diagram generator
├── test_pipeline.py           # Automated test suite (9 validation stages)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive documentation & viva defense guide
│
└── assets/
    └── architecture.png       # High-DPI Transformer text generation pipeline diagram
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE). Built for educational and research purposes.

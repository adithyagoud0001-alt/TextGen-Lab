"""
app.py - TextGen Lab: Interactive Transformer-Based Text Generation System

A complete educational Deep Learning & Generative AI web application built with:
- PyTorch & Hugging Face Transformers (DistilGPT-2)
- Streamlit UI
- Plotly interactive data visualizations
- NumPy numerical computing

Modules:
1. Interactive Text Generation
2. Generation Parameter Controls (Temperature, Top-K, Max Tokens, Repetition Penalty)
3. Next-Token Probability Visualization
4. Transformer Architecture Deep Dive & Interactive Pipeline
5. Temperature Sensitivity Experiment (T=0.3 vs 0.8 vs 1.3)
6. Mathematical Concepts (Softmax, Scaled Dot-Product Attention, Matrix Multiplication)
7. Viva Presentation & Student Evaluation Guide
"""

import os
import time
import streamlit as st
import numpy as np

# Import modular project components
from model import load_model_and_tokenizer, get_device, get_model_specifications
from generation import (
    generate_text,
    get_next_token_probabilities,
    run_temperature_experiment,
    compute_softmax,
)
from visualization import (
    plot_top_tokens,
    plot_temperature_effect_on_distribution,
    plot_softmax_mathematical_stages,
    create_architecture_diagram,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TextGen Lab – Transformer Text Generation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Styles & Typography */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    .badge-accent {
        background-color: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .output-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1.25rem;
        font-size: 1.02rem;
        line-height: 1.65;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .prompt-highlight {
        color: #1e293b;
        font-weight: 600;
        background-color: #e2e8f0;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
    }
    .generated-highlight {
        color: #1d4ed8;
        background-color: #eff6ff;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
    }
    .callout-box {
        border-left: 4px solid #6366f1;
        background-color: #f5f3ff;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #312e81;
    }
    .info-tag {
        font-size: 0.82rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = "Artificial intelligence will shape the future of"
if "generation_output" not in st.session_state:
    st.session_state.generation_output = None
if "next_token_data" not in st.session_state:
    st.session_state.next_token_data = None
if "temp_experiment_results" not in st.session_state:
    st.session_state.temp_experiment_results = None

# Ensure architecture diagram exists in assets/
os.makedirs("assets", exist_ok=True)
arch_img_path = os.path.join("assets", "architecture.png")
if not os.path.exists(arch_img_path):
    try:
        create_architecture_diagram(arch_img_path)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Model & Hardware Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_model():
    return load_model_and_tokenizer("distilgpt2")

with st.spinner("Initializing DistilGPT-2 and PyTorch runtime..."):
    tokenizer, model = get_cached_model()
    specs = get_model_specifications(model, tokenizer)

# -----------------------------------------------------------------------------
# Sidebar: Parameter Controls & Model Specifications
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")

    # Hardware & Model Specs
    st.markdown("### 🧩 Model Status")
    st.markdown(
        f"""
        - **Model:** `DistilGPT-2`
        - **Parameters:** `{specs['total_parameters_million']}M`
        - **Vocabulary Size:** `{specs['vocab_size']:,}`
        - **Device:** `{specs['device']}`
        - **Context Window:** `{specs['max_context_length']}` tokens
        """
    )
    st.markdown("---")

    # Module 2: Generation Parameter Controls
    st.markdown("### ⚙️ Generation Parameters")

    preset = st.selectbox(
        "Load Parameter Preset",
        [
            "Custom",
            "Balanced Creative (Default)",
            "Strict & Deterministic (Greedy)",
            "Highly Exploratory (High Temp)",
            "Focused Sampling (Top-K=10)",
        ],
        help="Quickly load tested hyperparameters for specific generation behaviors.",
    )

    # Preset logic
    default_max_tokens = 45
    default_temp = 0.8
    default_top_k = 50
    default_rep_pen = 1.15

    if preset == "Strict & Deterministic (Greedy)":
        default_temp = 0.0
        default_top_k = 0
        default_rep_pen = 1.1
    elif preset == "Highly Exploratory (High Temp)":
        default_temp = 1.3
        default_top_k = 80
        default_rep_pen = 1.1
    elif preset == "Focused Sampling (Top-K=10)":
        default_temp = 0.7
        default_top_k = 10
        default_rep_pen = 1.2

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=5,
        max_value=120,
        value=default_max_tokens,
        step=5,
        help="The maximum number of tokens to generate sequentially.",
    )

    temperature = st.slider(
        "Temperature (T)",
        min_value=0.0,
        max_value=2.0,
        value=default_temp,
        step=0.05,
        help="Scales model logits before Softmax. T=0 activates Greedy decoding (argmax). Higher T flattens probabilities.",
    )

    top_k = st.slider(
        "Top-K Sampling",
        min_value=0,
        max_value=100,
        value=default_top_k,
        step=5,
        help="Limits next-token candidate pool to the K most probable tokens. Set to 0 to disable Top-K.",
    )

    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=default_rep_pen,
        step=0.05,
        help="Penalizes tokens that have already been generated. Higher values reduce repetitive looping.",
    )

    use_seed = st.checkbox("Fix Random Seed", value=False, help="Ensures deterministic output for random sampling.")
    seed_val = st.number_input("Seed Value", value=42, step=1) if use_seed else None

    # Parameter Explanation Accordion
    with st.expander("ℹ️ How Parameters Work"):
        st.markdown(
            """
            - **Temperature (T):** Divides logits by $T$. When $T < 1.0$, differences between high and low logits are exaggerated (making the model confident). When $T > 1.0$, differences shrink, creating diverse or chaotic text.
            - **Top-K Sampling:** Truncates the probability distribution to only the top $K$ candidates before sampling, filtering out improbable tail tokens.
            - **Repetition Penalty:** Divides the positive logits of previously generated tokens, drastically reducing loops like "the same the same the same".
            - **Max New Tokens:** Caps the autoregressive generation loop to prevent runaway sequences.
            """
        )

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">TextGen Lab 🧪</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive Transformer-Based Text Generation & Generative AI Laboratory</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="badge-container">
        <span class="badge badge-accent">Pretrained Model: DistilGPT-2 (82M)</span>
        <span class="badge badge-green">Framework: PyTorch & Hugging Face</span>
        <span class="badge">Hardware Device: {specs['device']}</span>
        <span class="badge">Architecture: Decoder-Only Causal Transformer</span>
        <span class="badge">API Keys: None (100% Local Execution)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Tabs Navigation
# -----------------------------------------------------------------------------
tab_generation, tab_probs, tab_temp_exp, tab_architecture, tab_math, tab_viva = st.tabs(
    [
        "🚀 1. Text Generation",
        "📊 2. Next-Token Probabilities",
        "🌡️ 3. Temperature Experiment",
        "🧠 4. Transformer Architecture",
        "📐 5. Mathematical Concepts",
        "🎓 6. Student Viva Guide",
    ]
)

# =============================================================================
# TAB 1: INTERACTIVE TEXT GENERATION
# =============================================================================
with tab_generation:
    st.subheader("Module 1: Interactive Prompt & Generation")
    st.caption("Provide an input text prompt, configure generation controls in the sidebar, and watch DistilGPT-2 predict the continuation token-by-token.")

    # Sample prompt buttons
    st.markdown("**Sample Prompt Starters:**")
    sample_cols = st.columns(4)
    if sample_cols[0].button("🤖 AI & Society"):
        st.session_state.prompt_input = "Artificial intelligence will transform human society by"
    if sample_cols[1].button("🌌 Space & Science"):
        st.session_state.prompt_input = "To understand the laws of the universe, scientists must"
    if sample_cols[2].button("💻 Code & Computing"):
        st.session_state.prompt_input = "The fundamental difference between machine learning and traditional programming is"
    if sample_cols[3].button("📖 Creative Story"):
        st.session_state.prompt_input = "Once upon a time in a silent laboratory, the robotic assistant began to"

    # User Prompt Input Box
    prompt_text = st.text_area(
        "Input Text Prompt:",
        value=st.session_state.prompt_input,
        height=100,
        placeholder="Type a sentence or prompt to begin generating text...",
        help="Enter the opening prefix of text for the autoregressive model.",
    )
    st.session_state.prompt_input = prompt_text

    # Action Buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1, 4])
    with col_btn1:
        generate_clicked = st.button("🚀 Generate Text", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🔄 Clear Output", use_container_width=True)

    if clear_clicked:
        st.session_state.generation_output = None
        st.session_state.next_token_data = None
        st.rerun()

    if generate_clicked:
        if not prompt_text.strip():
            st.warning("⚠️ Please provide a non-empty text prompt before generating.")
        else:
            with st.spinner("Generating autoregressive continuation..."):
                gen_result = generate_text(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    seed=seed_val,
                )
                st.session_state.generation_output = gen_result

                # Also automatically compute next-token probabilities for the current prompt
                next_tok_res = get_next_token_probabilities(
                    tokenizer=tokenizer,
                    model=model,
                    text=prompt_text,
                    top_n=10,
                    temperature=max(temperature, 0.05) if temperature > 0 else 1.0,
                )
                st.session_state.next_token_data = next_tok_res

    # Display Generated Output
    if st.session_state.generation_output:
        res = st.session_state.generation_output
        st.markdown("### 📝 Model Continuation")

        st.markdown(
            f"""
            <div class="output-box">
                <span class="prompt-highlight">{res['prompt']}</span>
                <span class="generated-highlight">{res['generated_text']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Legend: Gray background indicates user prompt. Blue highlight indicates newly generated tokens.")

        # Performance & Generation Metrics
        st.markdown("#### ⏱️ Generation Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Prompt Tokens", res["num_prompt_tokens"])
        m_col2.metric("Generated Tokens", res["num_generated_tokens"])
        m_col3.metric("Latency", f"{res['generation_time_sec']}s")
        m_col4.metric("Throughput", f"{res['tokens_per_second']} tok/s")

        st.markdown(
            f"""
            > **Mode Active:** `{res['mode']}` | **Repetition Penalty:** `{repetition_penalty}`  
            > *Notice:* DistilGPT-2 is a statistical language model trained on WebText. It predicts likely word continuations based on learned patterns and does not guarantee factual correctness.
            """
        )

        # Quick probability snapshot preview
        if st.session_state.next_token_data and st.session_state.next_token_data.get("success"):
            top_tok = st.session_state.next_token_data["top_tokens"][0]
            st.info(
                f"💡 **Next-Token Insight:** For the prompt *'{prompt_text[-30:]}'*, the most probable next token is **'{top_tok['display_token']}'** with **{top_tok['percentage']}%** probability (Raw Logit: `{top_tok['logit']}`). Switch to Tab 2 to view the full probability distribution."
            )

# =============================================================================
# TAB 2: NEXT-TOKEN PROBABILITY VISUALIZATION
# =============================================================================
with tab_probs:
    st.subheader("Module 3: Next-Token Probability Distribution")
    st.caption("Inspect the exact mathematical probability distribution generated by DistilGPT-2's final Linear Head and Softmax layer for your prompt.")

    vis_prompt = st.text_input(
        "Text to Analyze Next Token Prediction:",
        value=st.session_state.prompt_input,
        help="The model will run a single forward pass over this sequence to predict the immediate next token.",
    )

    vis_col1, vis_col2, vis_col3 = st.columns([1.5, 1.5, 2])
    with vis_col1:
        top_n_select = st.radio("Top Candidates to Display:", [5, 10], horizontal=True, index=1)
    with vis_col2:
        vis_temp = st.slider("Preview with Temperature:", 0.1, 2.0, 1.0, 0.1, help="Observe how Softmax distribution flattens or sharpens with temperature.")
    with vis_col3:
        compute_probs_btn = st.button("📊 Calculate Next-Token Probabilities", use_container_width=True)

    if compute_probs_btn or (st.session_state.next_token_data is None and vis_prompt.strip()):
        with st.spinner("Executing forward pass and extracting output logits..."):
            res_probs = get_next_token_probabilities(
                tokenizer=tokenizer,
                model=model,
                text=vis_prompt,
                top_n=top_n_select,
                temperature=vis_temp,
            )
            st.session_state.next_token_data = res_probs

    if st.session_state.next_token_data and st.session_state.next_token_data.get("success"):
        tok_data = st.session_state.next_token_data
        top_list = tok_data["top_tokens"][:top_n_select]

        # Key metrics row
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Top-1 Predicted Token", tok_data["top_1_display"])
        p_col2.metric("Top-1 Probability", f"{tok_data['top_1_probability']}%")
        p_col3.metric("Vocabulary Probability Sum", f"{tok_data['sum_probabilities']:.4f}")
        p_col4.metric("Entropy (Uncertainty)", f"{tok_data['entropy']} bits")

        # Interactive Bar Chart
        fig_probs = plot_top_tokens(top_list, title=f"Top {top_n_select} Next-Token Candidates (T = {vis_temp})")
        st.plotly_chart(fig_probs, use_container_width=True)

        # Candidate breakdown table
        st.markdown("#### 📋 Detailed Candidate Token Breakdown")
        table_data = [
            {
                "Rank": item["rank"],
                "Token String": item["display_token"],
                "Token ID": item["token_id"],
                "Raw Logit ($z_i$)": item["logit"],
                "Softmax Probability ($P_i$)": f"{item['probability']:.4f}",
                "Percentage (%)": f"{item['percentage']}%",
            }
            for item in top_list
        ]
        st.dataframe(table_data, use_container_width=True)

        # Educational Explanation Callout
        st.markdown(
            """
            <div class="callout-box">
                <b>🎓 How the Model Selects the Next Token:</b><br>
                1. <b>Logits ($z$):</b> The final linear layer of DistilGPT-2 outputs an unnormalized vector of size <code>50,257</code> (one real number per vocabulary token).<br>
                2. <b>Softmax Transformation:</b> We compute $P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_{j} \\exp(z_j / T)}$ to map unconstrained logits to valid probabilities strictly in $[0, 1]$.<br>
                3. <b>Sum to 1.0:</b> Notice the vocabulary sum metric above is exactly <b>1.0000</b>, satisfying the fundamental axiom of probability distributions.<br>
                4. <b>Decoding Strategy:</b> If <i>Greedy decoding</i> is chosen, the model selects Rank #1 deterministically (argmax). If <i>Sampling</i> is enabled, the model rolls a random sample proportional to these percentage weights.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 3: TEMPERATURE SENSITIVITY EXPERIMENT
# =============================================================================
with tab_temp_exp:
    st.subheader("Module 5: Multi-Temperature Generation Experiment")
    st.caption("Witness firsthand how the Temperature hyperparameter alters token probability distributions and determines the creativity vs conservatism of text generation.")

    exp_prompt = st.text_input(
        "Experiment Prompt:",
        value=st.session_state.prompt_input,
        key="temp_exp_prompt_input",
    )

    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
    with exp_col1:
        exp_tokens = st.slider("Tokens per Generation", 15, 60, 35, 5)
    with exp_col2:
        exp_seed = st.number_input("Fixed Seed for Fair Comparison", value=42, step=1)
    with exp_col3:
        st.write("")
        st.write("")
        run_exp_btn = st.button("🧪 Run Multi-Temperature Experiment", type="primary", use_container_width=True)

    if run_exp_btn:
        with st.spinner("Generating side-by-side completions at T=0.3, T=0.8, and T=1.3..."):
            exp_data = run_temperature_experiment(
                tokenizer=tokenizer,
                model=model,
                prompt=exp_prompt,
                temperatures=[0.3, 0.8, 1.3],
                max_new_tokens=exp_tokens,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=int(exp_seed),
            )
            st.session_state.temp_experiment_results = exp_data

    if st.session_state.temp_experiment_results:
        exp_res = st.session_state.temp_experiment_results
        results = exp_res["results"]

        st.markdown("### 🔍 Side-by-Side Temperature Comparison")
        col_low, col_med, col_high = st.columns(3)

        cards = [col_low, col_med, col_high]
        for idx, (col, r) in enumerate(zip(cards, results)):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-top: 4px solid {r['color']};">
                        <h4 style="color: {r['color']}; margin-bottom: 0.3rem;">{r['title']}</h4>
                        <p style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.8rem;"><i>{r['description']}</i></p>
                        <div class="output-box" style="min-height: 180px; font-size: 0.95rem;">
                            <span class="prompt-highlight">{exp_res['prompt']}</span>
                            <span style="color: {r['color']}; font-weight: 500;">{r['generated_text']}</span>
                        </div>
                        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #475569;">
                            • <b>Tokens:</b> {r['num_tokens']}<br>
                            • <b>Latency:</b> {r['time_sec']}s ({r['tokens_per_sec']} tok/s)<br>
                            • <b>Lexical Diversity:</b> {r['lexical_diversity'] * 100:.1f}% unique words
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### 📊 Mathematical Distribution Reshaping")
        st.markdown("How the exact same top candidate tokens have their Softmax probabilities transformed across the 3 temperature regimes:")

        # Show probability reshaping chart
        with torch.no_grad():
            inp_enc = tokenizer(exp_prompt, return_tensors="pt").to(model.device)
            raw_logits = model(inp_enc["input_ids"]).logits[0, -1, :].cpu().numpy().astype(np.float64)
            top_ids = list(np.argsort(raw_logits)[-5:][::-1])

        fig_temp_compare = plot_temperature_effect_on_distribution(
            raw_logits=raw_logits,
            top_indices=top_ids,
            tokenizer=tokenizer,
            temperatures=[0.3, 0.8, 1.3],
        )
        st.plotly_chart(fig_temp_compare, use_container_width=True)

        st.markdown(
            """
            | Temperature Setting | Probability Distribution Shape | Characteristics | Best Use Cases |
            | :--- | :--- | :--- | :--- |
            | **Low ($T=0.3$)** | **Spiked & Peaked** (High confidence on top token) | Predictable, coherent, low vocabulary variety, susceptible to repetition loops. | Factual Q&A, math, code generation, summarization. |
            | **Medium ($T=0.8$)** | **Balanced Bell Curve** (Natural linguistic decay) | Coherent narrative flow with natural human-like lexical diversity. | Conversational AI, essays, general text generation. |
            | **High ($T=1.3$)** | **Flattened & Uniform** (High entropy / randomness) | Highly creative, novel vocabulary, but higher probability of hallucinations or ungrammatical phrases. | Brainstorming, poetry, surreal creative fiction. |
            """
        )

# =============================================================================
# TAB 4: TRANSFORMER ARCHITECTURE EXPLORER
# =============================================================================
with tab_architecture:
    st.subheader("Module 4: Transformer Architecture & Text Generation Pipeline")
    st.caption("A structural walk-through of how an autoregressive Decoder-Only Transformer processes raw text into next-token predictions.")

    # High-resolution Architecture Pipeline Diagram
    if os.path.exists(arch_img_path):
        st.image(arch_img_path, caption="Figure 1: Complete DistilGPT-2 Text Generation Pipeline Architecture", use_container_width=True)

    st.markdown("### 🧱 Step-by-Step Architecture Components")

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("1. Tokenization (Byte-Pair Encoding)", expanded=True):
            st.markdown(
                """
                - **Concept:** Neural networks cannot read characters or words directly. Text is split into subword units using **Byte-Pair Encoding (BPE)**.
                - **DistilGPT-2 Vocab Size:** `50,257` unique tokens.
                - **Prefix Space Notation:** Tokens with leading spaces often have distinct subword representations (e.g., `'Ġlearning'`).
                - **Formula:** $\\text{String} \\xrightarrow{\\text{BPE}} [\\text{ID}_1, \\text{ID}_2, \\dots, \\text{ID}_n]$
                """
            )

        with c2:
            with st.expander("2. Token & Positional Embeddings", expanded=True):
                st.markdown(
                    """
                - **Token Embedding ($W_e$):** Maps each discrete integer token ID into a continuous 768-dimensional latent vector.
                - **Positional Embedding ($W_p$):** Transformers have no recurrence (unlike RNNs/LSTMs). They learn a positional vector for positions $0$ to $1023$ to preserve word order.
                - **Combined Vector:** $X_0 = \\text{Embedding}(\\text{Token}) + \\text{Embedding}(\\text{Position})$
                """
                )

    c3, c4 = st.columns(2)
    with c3:
        with st.expander("3. Multi-Head Masked Self-Attention", expanded=True):
            st.markdown(
                """
                - **Self-Attention Mechanism:** Allows each token to attend to previous tokens in the sequence.
                - **Masking (Causal Attention):** Future tokens are masked out ($-\\infty$) so token $t$ cannot look ahead to token $t+1$.
                - **Multi-Head (12 Heads):** Each head projects input into distinct Query ($Q$), Key ($K$), and Value ($V$) subspaces of dimension $d_k = 768 / 12 = 64$.
                - **Scaled Dot-Product:**
                  $$\\text{Attention}(Q, K, V) = \\text{Softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
                """
            )

    with c4:
        with st.expander("4. Feed-Forward Network & LayerNorm", expanded=True):
            st.markdown(
                """
                - **Feed-Forward Network (FFN):** A two-layer perceptron with GELU non-linear activation.
                  $$\\text{FFN}(x) = \\text{GELU}(x W_1 + b_1) W_2 + b_2$$
                  Expands dimension from 768 to 3072, then projects back to 768.
                - **Residual Connections:** $x + \\text{Sublayer}(x)$ avoids vanishing gradients across deep layers.
                - **Layer Normalization:** Normalizes activations across features to stabilize gradient flow.
                """
            )

    c5, c6 = st.columns(2)
    with c5:
        with st.expander("5. Language Modeling Projection Head", expanded=True):
            st.markdown(
                """
                - **Projection:** Maps final hidden state of dimension `768` back up to the vocabulary dimension `50,257`.
                - **Weight Tying:** Often reuses the transpose of the input token embedding matrix ($W_e^T$).
                - **Output:** Raw unbounded log-odds scores (logits $z_i \\in \\mathbb{R}^{50,257}$).
                """
            )

    with c6:
        with st.expander("6. Autoregressive Loop", expanded=True):
            st.markdown(
                """
                - **Step 1:** Compute probabilities via Softmax.
                - **Step 2:** Sample or select the highest-probability token.
                - **Step 3:** Append selected token to the input prompt sequence:
                  $$\\text{Context}_{t+1} = [\\text{Context}_t, w_{\\text{new}}]$$
                - **Step 4:** Repeat forward pass until reaching `max_new_tokens` or the `<|endoftext|>` token.
                """
            )

# =============================================================================
# TAB 5: MATHEMATICAL CONCEPTS
# =============================================================================
with tab_math:
    st.subheader("Module 6: Mathematical Foundations of Generative AI")
    st.caption("Interactive equations, matrix operations, and numerical proofs behind modern language modeling.")

    st.markdown("### 🧮 1. The Softmax Function & Numerical Stability")
    st.markdown(
        """
        The Softmax function converts unnormalized model logits $z = [z_1, z_2, \\dots, z_K]$ into a valid probability distribution $P = [P_1, P_2, \\dots, P_K]$:
        $$\\text{Softmax}(z_i) = \\frac{\\exp(z_i / T)}{\\sum_{j=1}^{K} \\exp(z_j / T)}$$

        **Two Crucial Mathematical Properties:**
        1. **Strictly Non-Negative:** $\\exp(x) > 0 \\implies P_i > 0$ for all $i$.
        2. **Partition Normalization:** $\\sum_{i=1}^{K} P_i = 1.0$ (guarantees a legitimate probability distribution).
        
        **Numerical Stability Trick:** When calculating $\\exp(z)$, large logits (e.g. $z = 1000$) cause floating-point overflow. In practice, engineers subtract the maximum logit before exponentiation:
        $$\\exp(z_i - \\max(z))$$
        """
    )

    # Interactive Softmax Calculator
    st.markdown("#### 🔬 Interactive Softmax Playground")
    st.caption("Adjust the 4 sample logits below to observe live transformation into exponentiated scores and normalized probabilities:")

    sc1, sc2, sc3, sc4 = st.columns(4)
    l1 = sc1.slider("Logit 1 ('apple')", -5.0, 10.0, 4.2, 0.1)
    l2 = sc2.slider("Logit 2 ('banana')", -5.0, 10.0, 2.1, 0.1)
    l3 = sc3.slider("Logit 3 ('orange')", -5.0, 10.0, 0.5, 0.1)
    l4 = sc4.slider("Logit 4 ('grape')", -5.0, 10.0, -1.2, 0.1)

    fig_math_softmax = plot_softmax_mathematical_stages(
        sample_logits=[l1, l2, l3, l4],
        labels=["apple", "banana", "orange", "grape"],
    )
    st.plotly_chart(fig_math_softmax, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📐 2. Scaled Dot-Product Attention & Matrix Multiplication")
    st.markdown(
        """
        In self-attention, each input token vector is projected into three distinct representations via learned weight matrices:
        - **Query Matrix ($Q = X W_Q$):** What the current token is searching for.
        - **Key Matrix ($K = X W_K$):** What features the token contains to match against queries.
        - **Value Matrix ($V = X W_V$):** The actual semantic information transferred if attention aligns.

        #### The Attention Equation:
        $$\\text{Attention}(Q, K, V) = \\text{Softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} \\right) V$$

        #### Matrix Dimensional Breakdown:
        - Let sequence length be $N$, and hidden dimension per head be $d_k = 64$.
        - $Q \\in \\mathbb{R}^{N \\times d_k}$ and $K \\in \\mathbb{R}^{N \\times d_k}$.
        - $Q K^T \\in \\mathbb{R}^{N \\times N}$ is the **Attention Score Matrix**, measuring pairwise dot-product alignment between all tokens.
        - Dividing by $\\sqrt{d_k} = \\sqrt{64} = 8$ prevents the dot products from growing excessively large for large dimensions, which would push Softmax into regions with tiny gradients (gradient vanishing).
        - Multiplying by $V \\in \\mathbb{R}^{N \\times d_k}$ produces the context-aware output representation $\\in \\mathbb{R}^{N \\times d_k}$.
        """
    )

    st.markdown("---")
    st.markdown("### 🎲 3. Decoding & Sampling Strategies")
    st.markdown(
        """
        Once the probability distribution $P$ is computed across the $50,257$ vocabulary tokens, the system selects the next token using one of several decoding algorithms:
        
        1. **Greedy Search (Argmax):**
           $$w^* = \\arg\\max_{i} P_i$$
           Fast and deterministic, but frequently falls into repetitive loops and dull text.
           
        2. **Temperature Sampling:**
           $$P_i' = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$
           Randomly samples $w \\sim P_i'$. Lower $T$ increases certainty; higher $T$ increases diversity.
           
        3. **Top-K Filtering (Fan et al., 2018):**
           Sorts tokens by probability and retains only the top $K$ tokens. All other token probabilities are set to $0$, and the remaining $K$ probabilities are renormalized to sum to $1$. Prevents the model from sampling absurd or ungrammatical words from the long tail.
           
        4. **Repetition Penalty:**
           For each token $w$ that already appeared in the prompt or generated text:
           $$z_w = \\begin{cases} z_w / \\rho & \\text{if } z_w > 0 \\\\ z_w \\cdot \\rho & \\text{if } z_w < 0 \\end{cases}$$
           where $\\rho \\ge 1.0$ is the repetition penalty factor.
        """
    )

# =============================================================================
# TAB 6: STUDENT VIVA & PRESENTATION GUIDE
# =============================================================================
with tab_viva:
    st.subheader("Module 7: Student Viva Voce & College Defense Guide")
    st.caption("Essential revision questions, quick architectural summaries, and a 2-minute live demo script for college examinations and project presentations.")

    st.markdown(
        """
        ### ⏱️ 2-Minute Project Demonstration Script
        Follow this concise script when presenting TextGen Lab to examiners or professors:
        
        1. **Introduction (15s):**  
           *"Good morning! Today I am presenting **TextGen Lab**, an interactive deep learning system that demonstrates how modern Generative AI generates text using Transformer models. We use **DistilGPT-2**, an 82-million parameter autoregressive decoder-only Transformer loaded locally in PyTorch."*
        
        2. **Generation Demonstration (30s):**  
           *(Enter prompt 'Artificial intelligence will shape the future of' and click Generate)*  
           *"Here in Tab 1, the model processes our prompt and generates fluent continuation text at ~25 tokens per second on CPU. Notice the prompt is highlighted in gray and the generated tokens in blue."*
        
        3. **Next-Token Probabilities (30s):**  
           *(Switch to Tab 2)*  
           *"In Tab 2, we expose the internal mechanics. Before emitting any token, the final linear head outputs 50,257 raw logits. Softmax normalizes them into real probabilities summing exactly to 1.0. Here we visualize the top candidates with their exact probabilities."*
        
        4. **Temperature Sensitivity Experiment (30s):**  
           *(Switch to Tab 3 and show side-by-side results)*  
           *"In Tab 3, we demonstrate the mathematical role of Temperature. At T=0.3, the distribution is sharp and deterministic. At T=0.8, we get natural conversational variety. At T=1.3, the distribution flattens, increasing randomness and entropy."*
        
        5. **Conclusion & Architecture (15s):**  
           *"All computations run locally on PyTorch without external APIs. This proves the core principles of tokenization, self-attention, and autoregressive generation that power modern LLMs."*
        """
    )

    st.markdown("---")
    st.markdown("### ❓ Frequently Asked Viva Questions & Model Answers")

    q1, q2 = st.columns(2)
    with q1:
        with st.expander("Q1: Why did you choose DistilGPT-2 over full GPT-2 or LLaMA?", expanded=True):
            st.markdown(
                """
                **Answer:**  
                DistilGPT-2 uses **Knowledge Distillation** to compress GPT-2 Small from 124 million parameters down to **82 million parameters (6 layers vs 12)**. It retains over 95% of linguistic quality while running twice as fast and fitting effortlessly into local CPU memory without requiring expensive GPUs or paid cloud API keys.
                """
            )

        with q2:
            with st.expander("Q2: What is the mathematical difference between a logit and a probability?", expanded=True):
                st.markdown(
                    """
                **Answer:**  
                - **Logit ($z$):** An unconstrained real number ($-\\infty < z < +\\infty$) output directly by the model's final linear projection layer. It represents unnormalized log-odds.
                - **Probability ($P$):** A normalized metric strictly in the range $[0, 1]$, where the sum over all vocabulary tokens equals exactly $1.0$, obtained by applying the Softmax function to the logits.
                """
                )

    q3, q4 = st.columns(2)
    with q3:
        with st.expander("Q3: Why do we scale by $\\sqrt{d_k}$ in the Self-Attention formula?", expanded=True):
            st.markdown(
                """
                **Answer:**  
                For high dimensional vectors, dot products $Q K^T$ grow large in magnitude ($E[q \\cdot k] = 0, \\text{Var}[q \\cdot k] = d_k$). Large dot products push the Softmax function into saturation regions with extremely small gradients, causing vanishing gradients during backpropagation. Scaling by $\\frac{1}{\\sqrt{d_k}}$ keeps variance at 1.0.
                """
                )

    with q4:
        with st.expander("Q4: Why does DistilGPT-2 use Causal Masking?", expanded=True):
            st.markdown(
                """
                **Answer:**  
                DistilGPT-2 is an **autoregressive language model** designed for text generation. During training and inference, token $t$ must only attend to past tokens $1, 2, \\dots, t$. Causal masking places $-\\infty$ in the upper triangle of the attention matrix so $\\text{Softmax}(-\\infty) = 0$, ensuring the model cannot look ahead to future words.
                """
                )

    q5, q6 = st.columns(2)
    with q5:
        with st.expander("Q5: What is the difference between Greedy Search and Top-K Sampling?"):
            st.markdown(
                """
                **Answer:**  
                - **Greedy Search:** Always picks the token with the highest probability (argmax). It is deterministic, but often leads to repetitive loops.
                - **Top-K Sampling:** Selects the top $K$ highest-probability tokens, renormalizes their probabilities, and samples randomly from this restricted pool. This introduces controlled creativity while filtering out improbable junk tokens.
                """
            )

    with q6:
        with st.expander("Q6: Does this model guarantee factual accuracy?"):
            st.markdown(
                """
                **Answer:**  
                No. Transformer language models are statistical sequence predictors trained to maximize likelihood over a training corpus. They generate plausible-sounding text based on learned patterns and lack a ground-truth factual verification engine.
                """
            )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        <b>TextGen Lab</b> • Interactive Transformer Text Generation System • Designed for AI/ML Coursework & Viva Demonstration<br>
        Built with PyTorch, Hugging Face Transformers, DistilGPT-2, Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True,
)

"""
visualization.py - TextGen Lab Visualization & Charting Module

Provides:
- Interactive Plotly visualizations for next-token probability distributions.
- Multi-temperature distribution reshaping comparison charts.
- Step-by-step mathematical Softmax transformation visualizations.
- Scriptable high-resolution pipeline architecture diagram generation.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_top_tokens(
    top_tokens: List[Dict[str, Any]],
    title: str = "Top Predicted Next Tokens (Softmax Probabilities)",
) -> go.Figure:
    """
    Creates an interactive horizontal bar chart displaying candidate tokens,
    their probabilities, and highlights the #1 most probable next token.

    Args:
        top_tokens: List of token dicts (from generation.get_next_token_probabilities).
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    if not top_tokens:
        fig = go.Figure()
        fig.add_annotation(
            text="No token probability data available. Enter a prompt to generate.",
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
        )
        return fig

    # Invert so rank 1 appears at the top of the horizontal bar chart
    reversed_tokens = list(reversed(top_tokens))
    display_names = [item["display_token"] for item in reversed_tokens]
    percentages = [item["percentage"] for item in reversed_tokens]
    logits = [item["logit"] for item in reversed_tokens]
    token_ids = [item["token_id"] for item in reversed_tokens]
    ranks = [item["rank"] for item in reversed_tokens]

    # Highlight rank 1 token with distinct golden/emerald color
    colors = [
        "#10b981" if rank == 1 else "#6366f1"
        for rank in ranks
    ]

    hover_texts = [
        f"<b>Rank #{rank}: {name}</b><br>"
        f"Probability: <b>{pct:.2f}%</b><br>"
        f"Logit: <b>{lgt}</b><br>"
        f"Token ID: <code>{tid}</code>"
        for rank, name, pct, lgt, tid in zip(ranks, display_names, percentages, logits, token_ids)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=percentages,
                y=display_names,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="#1e1b4b", width=1),
                ),
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                hovertext=hover_texts,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color="#1e293b", family="sans-serif"),
        ),
        xaxis=dict(
            title="<b>Softmax Probability (%)</b>",
            ticksuffix="%",
            range=[0, max(max(percentages) * 1.15, 10)],
            gridcolor="#e2e8f0",
        ),
        yaxis=dict(
            title="<b>Candidate Token</b>",
            tickfont=dict(size=13, family="monospace"),
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        margin=dict(l=80, r=40, t=50, b=50),
        height=400,
    )

    return fig


def plot_temperature_effect_on_distribution(
    raw_logits: np.ndarray,
    top_indices: List[int],
    tokenizer: Any,
    temperatures: List[float] = [0.3, 0.8, 1.3],
) -> go.Figure:
    """
    Demonstrates how changing temperature alters the probability distribution
    over the exact same set of candidate tokens.

    Shows:
    - Low T: distribution peaks sharply around the argmax token.
    - Medium T: natural calibrated probability distribution.
    - High T: distribution flattens, increasing randomness/entropy.

    Args:
        raw_logits: 1D numpy array of vocabulary logits.
        top_indices: List of candidate token IDs to visualize.
        tokenizer: PreTrainedTokenizer for decoding token IDs.
        temperatures: List of temperature values.

    Returns:
        Plotly Figure showing grouped comparative bars.
    """
    token_labels = []
    for idx in top_indices:
        token_str = tokenizer.decode([idx])
        display_tok = repr(token_str).strip("'")
        if token_str == " ":
            display_tok = "␣ (space)"
        elif token_str == "\n":
            display_tok = "\\n"
        token_labels.append(display_tok)

    color_map = {
        0.3: ("#2563eb", "Low T (0.3) - Sharp & Focused"),
        0.8: ("#16a34a", "Medium T (0.8) - Balanced"),
        1.3: ("#dc2626", "High T (1.3) - Flattened & High Entropy"),
    }

    fig = go.Figure()

    for temp in temperatures:
        # Scale logits
        scaled = raw_logits / max(temp, 1e-4)
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        all_probs = exp_vals / np.sum(exp_vals)
        selected_probs = [float(all_probs[idx]) * 100 for idx in top_indices]

        color, label = color_map.get(temp, ("#4b5563", f"T={temp}"))

        fig.add_trace(
            go.Bar(
                name=label,
                x=token_labels,
                y=selected_probs,
                marker=dict(color=color),
                text=[f"{p:.1f}%" for p in selected_probs],
                textposition="outside",
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>How Temperature Reshapes Next-Token Probabilities</b>",
            font=dict(size=17, color="#1e293b"),
        ),
        barmode="group",
        xaxis=dict(
            title="<b>Candidate Tokens</b>",
            tickfont=dict(family="monospace", size=13),
        ),
        yaxis=dict(
            title="<b>Probability (%)</b>",
            ticksuffix="%",
            gridcolor="#e2e8f0",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        margin=dict(l=50, r=30, t=70, b=50),
        height=400,
    )

    return fig


def plot_softmax_mathematical_stages(
    sample_logits: List[float],
    labels: List[str],
) -> go.Figure:
    """
    Renders a multi-panel visual demonstration of the three stages of Softmax:
    Stage 1: Raw Logits (-inf to +inf)
    Stage 2: Exponentiated Scores (exp(x) >= 0)
    Stage 3: Normalized Probabilities (sum = 100%)

    Args:
        sample_logits: List of numerical logits.
        labels: String labels corresponding to each logit.

    Returns:
        Plotly Figure with 3 subplots.
    """
    logits_arr = np.array(sample_logits, dtype=np.float64)
    # Step 2: Exp
    shifted = logits_arr - np.max(logits_arr)
    exp_scores = np.exp(shifted)
    # Step 3: Normalized
    probs = (exp_scores / np.sum(exp_scores)) * 100

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "<b>1. Raw Logits</b><br><i>zᵢ ∈ (-∞, +∞)</i>",
            "<b>2. Exponentiation</b><br><i>exp(zᵢ - max) ≥ 0</i>",
            "<b>3. Normalized Probabilities</b><br><i>Pᵢ = exp(zᵢ) / Σ exp(zⱼ)</i>",
        ),
    )

    # 1. Raw Logits
    fig.add_trace(
        go.Bar(
            x=labels,
            y=sample_logits,
            marker=dict(color="#3b82f6"),
            text=[f"{v:.1f}" for v in sample_logits],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Exponentiated
    fig.add_trace(
        go.Bar(
            x=labels,
            y=exp_scores,
            marker=dict(color="#f59e0b"),
            text=[f"{v:.2f}" for v in exp_scores],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Softmax Probabilities
    fig.add_trace(
        go.Bar(
            x=labels,
            y=probs,
            marker=dict(color="#10b981"),
            text=[f"{v:.1f}%" for v in probs],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=dict(
            text="<b>Mathematical Transformation: Raw Logits → Softmax Probabilities</b>",
            font=dict(size=16, color="#0f172a"),
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=360,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    fig.update_yaxes(title_text="Logit Score", row=1, col=1, gridcolor="#e2e8f0")
    fig.update_yaxes(title_text="exp(z) Value", row=1, col=2, gridcolor="#e2e8f0")
    fig.update_yaxes(title_text="Probability (%)", ticksuffix="%", row=1, col=3, gridcolor="#e2e8f0")

    return fig


def create_architecture_diagram(output_path: str = "assets/architecture.png") -> str:
    """
    Generates a high-resolution, publication-quality architecture diagram of
    the Transformer-based Text Generation pipeline using Matplotlib.

    Pipeline:
    Input Text → Tokenization (BPE) → Token + Positional Embeddings →
    Transformer Blocks (Multi-Head Self-Attention + FFN) → LayerNorm →
    Linear Projection Head (Vocab 50,257) → Softmax → Next Token Selection

    Args:
        output_path: File path where image should be saved.

    Returns:
        Absolute or relative path to saved diagram image.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Title & Subtitle
    ax.text(
        7.5, 8.0,
        "DistilGPT-2 Autoregressive Text Generation Pipeline",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="#0f172a"
    )
    ax.text(
        7.5, 7.6,
        "End-to-End Deep Learning Architecture: From Prompt Tokens to Next-Token Probability Distribution",
        ha="center", va="center",
        fontsize=11, color="#475569"
    )

    boxes = [
        {
            "id": 1, "x": 0.7, "y": 4.8, "w": 2.5, "h": 2.2,
            "title": "1. Input Text",
            "body": "User prompt prefix:\n\"Artificial intelligence\nwill transform...\"",
            "color": "#e0e7ff", "border": "#4338ca",
        },
        {
            "id": 2, "x": 3.8, "y": 4.8, "w": 2.6, "h": 2.2,
            "title": "2. Tokenizer (BPE)",
            "body": "Byte-Pair Encoding\nVocab Size: 50,257\nTokens: [\"Art\", \"ificial\", ...]\nToken IDs: [7928, 4430...]",
            "color": "#dbeafe", "border": "#1d4ed8",
        },
        {
            "id": 3, "x": 7.0, "y": 4.8, "w": 2.6, "h": 2.2,
            "title": "3. Embeddings",
            "body": "Token Embeddings (768d)\n+\nLearned Positional\nEmbeddings (0 to 1023)\nX_0 = E_tok + E_pos",
            "color": "#cffafe", "border": "#0e7490",
        },
        {
            "id": 4, "x": 10.2, "y": 4.6, "w": 3.9, "h": 2.6,
            "title": "4. Transformer Blocks (x6 Layers)",
            "body": "• Masked Multi-Head Attention (12 Heads)\n  Q, K, V Projections (d_k = 64)\n  Attention = Softmax(QK^T / √d_k) V\n• Residual Add & LayerNorm\n• Feed-Forward Network (768 → 3072 → 768)\n• Residual Add & LayerNorm",
            "color": "#fef3c7", "border": "#b45309",
        },
        {
            "id": 5, "x": 10.4, "y": 1.3, "w": 3.5, "h": 2.2,
            "title": "5. LM Projection Head",
            "body": "Final LayerNorm\nLinear Projection: R^768 → R^50,257\nRaw Unbounded Logits:\nz = [z_1, z_2, ..., z_50257]",
            "color": "#fce7f3", "border": "#be185d",
        },
        {
            "id": 6, "x": 6.3, "y": 1.3, "w": 3.4, "h": 2.2,
            "title": "6. Softmax & Sampling",
            "body": "P(w_i) = exp(z_i / T) / Σ exp(z_j / T)\n• Temperature (T) Scaling\n• Top-K Candidate Truncation\n• Repetition Penalty Filtering",
            "color": "#dcfce7", "border": "#15803d",
        },
        {
            "id": 7, "x": 1.5, "y": 1.3, "w": 4.1, "h": 2.2,
            "title": "7. Next Token & Autoregression",
            "body": "Selected Token: \" future\"\nAppends to Sequence:\nPrompt ← Prompt + \" future\"\nRepeats loop until EOS / Max Tokens",
            "color": "#ede9fe", "border": "#6d28d9",
        },
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.1,rounding_size=0.15",
            linewidth=2, edgecolor=b["border"], facecolor=b["color"],
        )
        ax.add_patch(rect)
        # Header title
        ax.text(
            b["x"] + b["w"] / 2, b["y"] + b["h"] - 0.35,
            b["title"],
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#0f172a"
        )
        # Body text
        ax.text(
            b["x"] + b["w"] / 2, b["y"] + (b["h"] - 0.45) / 2,
            b["body"],
            ha="center", va="center",
            fontsize=8.5, color="#1e293b", family="sans-serif",
            linespacing=1.35,
        )

    # Connecting arrows
    arrow_props = dict(
        facecolor="#475569", edgecolor="#475569",
        width=1.8, headwidth=7.5, headlength=7.5, shrink=0.04
    )

    # 1 -> 2
    ax.annotate("", xy=(3.8, 5.9), xytext=(3.2, 5.9), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(7.0, 5.9), xytext=(6.4, 5.9), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate("", xy=(10.2, 5.9), xytext=(9.6, 5.9), arrowprops=arrow_props)
    # 4 down to 5
    ax.annotate(
        "", xy=(12.15, 3.5), xytext=(12.15, 4.6),
        arrowprops=dict(
            facecolor="#be185d", edgecolor="#be185d",
            width=2.0, headwidth=8, headlength=8, shrink=0.04
        )
    )
    # 5 -> 6
    ax.annotate(
        "", xy=(9.7, 2.4), xytext=(10.4, 2.4),
        arrowprops=arrow_props
    )
    # 6 -> 7
    ax.annotate(
        "", xy=(5.6, 2.4), xytext=(6.3, 2.4),
        arrowprops=arrow_props
    )
    # 7 -> 1 (Autoregressive Feedback Loop)
    ax.annotate(
        "Autoregressive Loop\n(Append Token to Prompt)",
        xy=(0.7, 5.9), xytext=(0.4, 3.6),
        ha="center", va="center",
        fontsize=9.0, fontweight="bold", color="#6d28d9",
        arrowprops=dict(
            facecolor="#6d28d9", edgecolor="#6d28d9",
            width=2.2, headwidth=8.5, headlength=8.5,
            connectionstyle="arc3,rad=-0.4", shrink=0.05
        )
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path

"""
model.py - TextGen Lab Model Management Module

Handles:
- Loading and caching the DistilGPT-2 model and tokenizer.
- Automatic hardware device detection (GPU/CUDA or CPU).
- Extraction of model architecture specifications for educational inspection.
"""

from typing import Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st


def get_device() -> torch.device:
    """
    Detects and returns the optimal available computation device.
    Uses CUDA GPU if available; otherwise defaults gracefully to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource(show_spinner="Loading DistilGPT-2 model & tokenizer into memory...")
def load_model_and_tokenizer(model_name: str = "distilgpt2") -> Tuple[Any, Any]:
    """
    Loads pretrained tokenizer and Causal Language Model from Hugging Face Hub.
    Cached via Streamlit resource caching to ensure loading happens only once.

    Args:
        model_name: Hugging Face model identifier (default: "distilgpt2").

    Returns:
        Tuple of (tokenizer, model).
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # DistilGPT-2 has no explicit pad token; standard practice is to use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout)

    # Align model pad token ID
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def get_model_specifications(model: Any, tokenizer: Any) -> Dict[str, Any]:
    """
    Extracts structural architectural specifications of DistilGPT-2
    to explain its deep learning parameters to students.

    Args:
        model: Hugging Face PreTrainedModel instance.
        tokenizer: Hugging Face PreTrainedTokenizer instance.

    Returns:
        Dictionary of architectural metrics and hyperparameter specs.
    """
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": config.name_or_path if hasattr(config, "name_or_path") else "distilgpt2",
        "total_parameters": total_params,
        "total_parameters_million": round(total_params / 1e6, 2),
        "trainable_parameters": trainable_params,
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": getattr(config, "n_embd", 768),
        "num_layers": getattr(config, "n_layer", 6),
        "num_heads": getattr(config, "n_head", 12),
        "max_context_length": getattr(config, "n_positions", 1024),
        "device": str(get_device()).upper(),
        "architecture_type": "Decoder-Only Autoregressive Transformer",
    }

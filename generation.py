"""
generation.py - TextGen Lab Text Generation & Probability Computation Module

Handles:
- Autoregressive text generation with configurable sampling parameters (Temperature, Top-K, Repetition Penalty).
- Extracting raw model logits and converting them to valid probability distributions via Softmax.
- Multi-temperature comparative generation for side-by-side diversity experiments.
- Numerical validation of Softmax properties (summing to 1.0).
"""

import time
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Generates text continuation from a prompt using DistilGPT-2.

    Args:
        tokenizer: PreTrainedTokenizer instance.
        model: PreTrainedModel instance.
        prompt: User input string.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (>0.0). If <=0.0, greedy decoding is used.
        top_k: Top-K filter threshold (>0).
        repetition_penalty: Multiplier penalizing repeated tokens (>1.0 reduces loops).
        seed: Optional integer seed for reproducible generation.

    Returns:
        Dictionary with generated text, metrics, and execution metadata.
    """
    clean_prompt = prompt.strip() if prompt else ""
    if not clean_prompt:
        return {
            "success": False,
            "error": "Input prompt is empty. Please enter text to generate a continuation.",
            "prompt": "",
            "generated_text": "",
            "full_text": "",
            "num_prompt_tokens": 0,
            "num_generated_tokens": 0,
            "generation_time_sec": 0.0,
            "tokens_per_second": 0.0,
        }

    device = model.device

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Encode input prompt
    inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    # Configure decoding strategy
    is_greedy = (temperature <= 0.0)
    effective_temp = max(float(temperature), 1e-4) if not is_greedy else 1.0
    effective_top_k = int(top_k) if (top_k > 0 and not is_greedy) else None

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": float(repetition_penalty),
    }

    if is_greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = effective_temp
        if effective_top_k:
            gen_kwargs["top_k"] = effective_top_k

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs,
        )
    elapsed_time = max(time.perf_counter() - start_time, 1e-6)

    # Extract only newly generated tokens
    new_token_ids = output_ids[0][prompt_tokens:]
    num_generated_tokens = len(new_token_ids)
    tokens_per_sec = round(num_generated_tokens / elapsed_time, 2)

    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "success": True,
        "error": None,
        "prompt": clean_prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "num_prompt_tokens": prompt_tokens,
        "num_generated_tokens": num_generated_tokens,
        "generation_time_sec": round(elapsed_time, 3),
        "tokens_per_second": tokens_per_sec,
        "mode": "Greedy (Deterministic)" if is_greedy else f"Sampling (T={effective_temp}, Top-K={top_k})",
    }


def compute_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable Softmax calculation:
    Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Subtracting the maximum logit prevents floating-point overflow in exp().

    Args:
        logits: 1D array of unnormalized log-odds scores.
        temperature: Temperature scaling factor.

    Returns:
        1D array of normalized probabilities summing to 1.0.
    """
    t = max(temperature, 1e-4)
    scaled_logits = logits / t
    # Numerical stability shift
    shifted_logits = scaled_logits - np.max(scaled_logits)
    exp_scores = np.exp(shifted_logits)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities


def get_next_token_probabilities(
    tokenizer: Any,
    model: Any,
    text: str,
    top_n: int = 10,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Performs a single forward pass over input text to extract real model logits
    for the next token position, calculates Softmax probabilities, and ranks
    the top candidates.

    Args:
        tokenizer: PreTrainedTokenizer instance.
        model: PreTrainedModel instance.
        text: Input string.
        top_n: Number of candidate tokens to return (default: 10).
        temperature: Scaling factor for probability sharpening/flattening.

    Returns:
        Dictionary containing:
        - top_tokens: List of dicts (token, token_id, logit, probability, percentage).
        - sum_probabilities: Total sum across entire vocabulary (~1.0).
        - entropy: Shannon entropy of next-token distribution (measure of uncertainty).
        - top_1_token: The greedy/argmax next token.
    """
    clean_text = text.strip() if text else ""
    if not clean_text:
        return {
            "success": False,
            "error": "Input text is required to calculate next token predictions.",
            "top_tokens": [],
            "sum_probabilities": 0.0,
            "entropy": 0.0,
            "top_1_token": "",
        }

    device = model.device
    inputs = tokenizer(clean_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        # Logits of shape [batch_size=1, sequence_length, vocab_size]
        # We take the last sequence position: [vocab_size]
        last_token_logits = outputs.logits[0, -1, :]

    # Convert to CPU numpy array for transparent demonstration
    raw_logits = last_token_logits.cpu().numpy().astype(np.float64)

    # Calculate probabilities using numerically stable Softmax
    probs = compute_softmax(raw_logits, temperature=temperature)
    total_prob_sum = float(np.sum(probs))

    # Calculate Shannon entropy (H = -sum(p * log2(p))) as measure of prediction certainty
    # Avoid log(0) using small epsilon
    eps = 1e-12
    entropy = -float(np.sum(probs * np.log2(probs + eps)))

    # Get top-N candidate indices
    top_indices = np.argsort(probs)[-top_n:][::-1]

    top_tokens = []
    for rank, idx in enumerate(top_indices, start=1):
        token_str = tokenizer.decode([idx])
        # Make whitespace and special chars visible to students
        display_token = repr(token_str).strip("'")
        if token_str == " ":
            display_token = "␣ (space)"
        elif token_str == "\n":
            display_token = "\\n (newline)"

        prob_val = float(probs[idx])
        logit_val = float(raw_logits[idx])

        top_tokens.append({
            "rank": rank,
            "token": token_str,
            "display_token": display_token,
            "token_id": int(idx),
            "logit": round(logit_val, 3),
            "probability": prob_val,
            "percentage": round(prob_val * 100, 2),
        })

    return {
        "success": True,
        "error": None,
        "input_text": clean_text,
        "temperature": temperature,
        "top_tokens": top_tokens,
        "sum_probabilities": round(total_prob_sum, 6),
        "entropy": round(entropy, 3),
        "top_1_token": top_tokens[0]["token"] if top_tokens else "",
        "top_1_display": top_tokens[0]["display_token"] if top_tokens else "",
        "top_1_probability": top_tokens[0]["percentage"] if top_tokens else 0.0,
        "vocab_size": len(raw_logits),
    }


def run_temperature_experiment(
    tokenizer: Any,
    model: Any,
    prompt: str,
    temperatures: List[float] = [0.3, 0.8, 1.3],
    max_new_tokens: int = 40,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs text generation across three distinct temperature regimes:
    - Low (0.3): Conservative, focused, deterministic tendencies.
    - Medium (0.8): Balanced coherence and linguistic diversity.
    - High (1.3): Highly creative, exploratory, potentially erratic.

    Args:
        tokenizer: PreTrainedTokenizer instance.
        model: PreTrainedModel instance.
        prompt: User input string.
        temperatures: List of three temperature float values.
        max_new_tokens: Number of tokens to generate per experiment.
        top_k: Top-k filter.
        repetition_penalty: Repetition penalty factor.
        seed: Fixed seed to isolate temperature as the primary independent variable.

    Returns:
        Dictionary mapping temperature labels to generation results and analysis.
    """
    labels = {
        0.3: ("Low Temperature (T = 0.3)", "Conservative & Deterministic", "#2563eb"),
        0.8: ("Medium Temperature (T = 0.8)", "Balanced & Natural", "#16a34a"),
        1.3: ("High Temperature (T = 1.3)", "Creative & High Entropy", "#dc2626"),
    }

    results = []
    for temp in temperatures:
        label_info = labels.get(
            temp,
            (f"Temperature (T = {temp})", "Custom Regime", "#4b5563")
        )
        gen_result = generate_text(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        # Lexical diversity metric: Unique words / Total words
        words = gen_result["generated_text"].split()
        unique_words = len(set(words))
        total_words = max(len(words), 1)
        diversity_ratio = round(unique_words / total_words, 3)

        results.append({
            "temperature": temp,
            "title": label_info[0],
            "description": label_info[1],
            "color": label_info[2],
            "generated_text": gen_result["generated_text"],
            "full_text": gen_result["full_text"],
            "num_tokens": gen_result["num_generated_tokens"],
            "time_sec": gen_result["generation_time_sec"],
            "tokens_per_sec": gen_result["tokens_per_second"],
            "lexical_diversity": diversity_ratio,
        })

    return {
        "prompt": prompt,
        "seed": seed,
        "results": results,
    }

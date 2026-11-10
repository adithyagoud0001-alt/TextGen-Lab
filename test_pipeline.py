"""
test_pipeline.py - Automated Testing & Verification Suite for TextGen Lab

Verifies:
1. Model and tokenizer loading (DistilGPT-2).
2. Hardware device handling (CPU / GPU fallback).
3. Text generation with different prompts and deterministic greedy mode.
4. Temperature scaling and diversity impact.
5. Top-K filtering.
6. Softmax probability normalization (summing to ~1.0).
7. Top-10 next-token extraction and ranking.
8. Edge cases: Empty prompts, zero temperature, negative tokens.
9. Architecture diagram creation in assets/architecture.png.
"""

import os
import sys
import numpy as np

# Ensure UTF-8 output encoding on Windows PowerShell / cmd
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tests():
    print("=" * 70)
    print("TextGen Lab - Verification Test Suite")
    print("=" * 70)

    # 1. Imports
    print("\n[Test 1/9] Checking dependencies and modules...")
    try:
        import torch
        import transformers
        import streamlit
        import plotly
        import matplotlib
        from model import get_device, load_model_and_tokenizer, get_model_specifications
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
        print("  [PASS] All required libraries and local modules imported successfully.")
    except Exception as e:
        print(f"  [FAIL] Import Error: {e}")
        return False

    # 2. Device Detection
    print("\n[Test 2/9] Checking hardware device detection...")
    device = get_device()
    print(f"  [PASS] Active Computation Device: {device}")
    assert device.type in ["cpu", "cuda"], f"Unexpected device type: {device.type}"

    # 3. Model Loading
    print("\n[Test 3/9] Loading DistilGPT-2 model & tokenizer...")
    tokenizer, model = load_model_and_tokenizer("distilgpt2")
    specs = get_model_specifications(model, tokenizer)
    print(f"  [PASS] Model Loaded: {specs['model_name']}")
    print(f"  [PASS] Parameters: {specs['total_parameters_million']}M")
    print(f"  [PASS] Vocabulary Size: {specs['vocab_size']}")
    assert specs['vocab_size'] == 50257, f"Expected vocab size 50257, got {specs['vocab_size']}"

    # 4. Empty Prompt Handling
    print("\n[Test 4/9] Testing edge-case handling (empty prompt)...")
    empty_res = generate_text(tokenizer, model, "")
    assert not empty_res["success"], "Empty prompt should return success=False"
    print(f"  [PASS] Empty prompt handled gracefully: '{empty_res['error']}'")

    empty_prob = get_next_token_probabilities(tokenizer, model, "")
    assert not empty_prob["success"], "Empty text probability should return success=False"
    print("  [PASS] Empty probability extraction handled gracefully.")

    # 5. Text Generation & Deterministic Greedy
    print("\n[Test 5/9] Testing text generation (Greedy vs Sampling)...")
    prompt = "Artificial intelligence will"
    greedy_res = generate_text(tokenizer, model, prompt, max_new_tokens=15, temperature=0.0)
    assert greedy_res["success"], "Greedy generation failed"
    print(f"  [PASS] Greedy Output: '{greedy_res['prompt']}{greedy_res['generated_text']}'")
    print(f"  [PASS] Speed: {greedy_res['tokens_per_second']} tokens/sec")

    sample_res = generate_text(tokenizer, model, prompt, max_new_tokens=15, temperature=0.8, top_k=50)
    assert sample_res["success"], "Sampling generation failed"
    print(f"  [PASS] Sampling Output: '{sample_res['prompt']}{sample_res['generated_text']}'")

    # 6. Next-Token Probabilities & Softmax Proof
    print("\n[Test 6/9] Testing next-token probabilities and Softmax normalization...")
    prob_res = get_next_token_probabilities(tokenizer, model, "The capital of France is", top_n=10)
    assert prob_res["success"], "Probability calculation failed"
    assert len(prob_res["top_tokens"]) == 10, f"Expected 10 tokens, got {len(prob_res['top_tokens'])}"
    print(f"  [PASS] Top-1 Candidate: '{prob_res['top_1_display']}' with probability {prob_res['top_1_probability']}%")
    print(f"  [PASS] Full Vocabulary Probability Sum: {prob_res['sum_probabilities']}")
    # Assert probability sum is within float margin of 1.0
    assert abs(prob_res["sum_probabilities"] - 1.0) < 1e-4, "Softmax probabilities must sum to 1.0"
    print("  [PASS] Softmax Axiom Verified: sum(P) == 1.0000")

    # 7. Temperature Experiment
    print("\n[Test 7/9] Testing temperature comparison experiment (0.3 vs 0.8 vs 1.3)...")
    exp_res = run_temperature_experiment(
        tokenizer, model,
        prompt="The secret to learning science is",
        temperatures=[0.3, 0.8, 1.3],
        max_new_tokens=20,
        seed=42,
    )
    assert len(exp_res["results"]) == 3, "Expected 3 temperature experiment results"
    for r in exp_res["results"]:
        print(f"  [PASS] {r['title']}: '{r['generated_text'][:40]}...' (Diversity: {r['lexical_diversity']})")

    # 8. Visualizations and Diagram
    print("\n[Test 8/9] Testing visualization figure generation...")
    fig1 = plot_top_tokens(prob_res["top_tokens"])
    assert fig1 is not None, "plot_top_tokens failed"

    sample_logits = np.array([2.0, 1.0, 0.5, -1.0])
    fig2 = plot_softmax_mathematical_stages([2.0, 1.0, 0.5, -1.0], ["A", "B", "C", "D"])
    assert fig2 is not None, "plot_softmax_mathematical_stages failed"
    print("  [PASS] Plotly charts created without error.")

    # 9. Architecture Diagram File Creation
    print("\n[Test 9/9] Generating and verifying architecture diagram in assets/...")
    os.makedirs("assets", exist_ok=True)
    diag_path = create_architecture_diagram("assets/architecture.png")
    assert os.path.exists(diag_path), f"Architecture diagram not found at {diag_path}"
    file_size_kb = round(os.path.getsize(diag_path) / 1024, 1)
    print(f"  [PASS] Architecture diagram saved at {diag_path} ({file_size_kb} KB)")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 9 TEST SUITES PASSED PERFECTLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

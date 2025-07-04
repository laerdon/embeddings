#!/usr/bin/env python3
"""Test embedding distance after changing a single character.

Example:
    python single_char_test.py --pdf paper.pdf --backend openai --pos 100 --new-char X
"""

import argparse
from embedding_collision_test import EmbeddingCollisionTester
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def replace_char(text: str, position: int, new_char: str) -> str:
    if position < 0 or position >= len(text):
        raise ValueError("position out of range")
    return text[:position] + new_char + text[position + 1 :]


def main():
    parser = argparse.ArgumentParser(description="single-character perturbation test")
    parser.add_argument("--pdf", type=str, required=True, help="path to PDF file")
    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-embedding-3-small",
        help="openai embedding model name",
    )
    parser.add_argument(
        "--pos", type=int, default=0, help="character index to change (0-based)"
    )
    parser.add_argument(
        "--new-char", type=str, default="X", help="replacement character (single char)"
    )
    args = parser.parse_args()

    tester = EmbeddingCollisionTester(
        backend="openai", model_name=args.openai_model, embed_full=True
    )
    text = tester.extract_text_from_pdf(args.pdf)
    if not text:
        print("[error] could not extract text from pdf")
        return

    canonical = " ".join(text.split())
    if args.pos >= len(canonical):
        print(f"[error] position {args.pos} exceeds text length {len(canonical)}")
        return

    variant = replace_char(canonical, args.pos, args.new_char)

    emb_orig = tester.get_embedding(canonical)
    emb_var = tester.get_embedding(variant)

    euclid = np.linalg.norm(emb_orig - emb_var)
    cos_sim = cosine_similarity([emb_orig], [emb_var])[0][0]

    print("single-character perturbation result")
    print("= " * 25)
    print(f"char position: {args.pos}")
    print(f"original char: '{canonical[args.pos]}' → new char: '{args.new_char}'")
    print(f"euclidean distance: {euclid:.6f}")
    print(f"cosine similarity:  {cos_sim:.6f}")


if __name__ == "__main__":
    main()

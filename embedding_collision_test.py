#!/usr/bin/env python3
"""
BERT Embedding Collision Test

Research question: If we embed a large text and systematically change one word,
at what point do we get identical embeddings (collisions)?

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
from typing import List, Dict, Tuple
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from datetime import datetime

# openai backend (>=1.0 syntax)
try:
    import openai  # type: ignore
    from openai import OpenAI as _OpenAIClient  # type: ignore
    import tiktoken  # type: ignore
except ImportError:
    openai = None  # will raise if backend requested
    _OpenAIClient = None
    tiktoken = None


# ---------------------------------------------
# embedding collision tester
# ---------------------------------------------


class EmbeddingCollisionTester:
    """Test for embedding collisions when changing words in a text.

    New in this revision:
    1. Optionally embed the *entire* text by averaging embeddings of
       sequential 512-token windows (instead of truncating).
    2. Ability to flip *k* words per variant (not just one) to probe
       how many changes are needed before collisions disappear.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embed_full: bool = False,
        backend: str = "bert",
        openai_api_key: str = None,
    ):
        """Initialize BERT model and tokenizer.

        Args:
            model_name: huggingface model name
            embed_full: if True, embed the *entire* text by sliding a
                         512-token window and mean-pooling the resulting
                         segment embeddings. if False, we keep the
                         original behaviour of truncating to 512 tokens.
            backend: embedding backend to use
            openai_api_key: API key for OpenAI backend
        """
        self.backend = backend.lower()
        self.embed_full = embed_full

        # collision thresholds
        self.collision_threshold = 1e-6
        self.near_collision_threshold = 1e-4

        if self.backend == "bert":
            print(f"[debug] backend: bert ({model_name})")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[debug] using device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        elif self.backend == "openai":
            if openai is None:
                raise ImportError(
                    "openai and tiktoken packages are required for openai backend"
                )
            print(f"[debug] backend: openai ({model_name})")
            openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable not set and no key provided"
                )

            self.openai_model = (
                model_name  # e.g., 'text-embedding-3-small' or 'text-embedding-ada-002'
            )

            # instantiate client once (thread-safe per openai docs)
            self._oa_client = _OpenAIClient(api_key=openai.api_key)

            # tiktoken encoding for token counting
            try:
                self.encoding = tiktoken.encoding_for_model(self.openai_model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")

            # context window (openai doc: embeddings models support 8191 tokens)
            self.openai_max_tokens = 8191
        else:
            raise ValueError(
                f"unsupported backend '{backend}'. choose 'bert' or 'openai'."
            )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + " "
        except Exception as e:
            print(f"[error] reading pdf: {e}")
            return ""
        return text.strip()

    # -------------------------------------------------
    # embedding helpers
    # -------------------------------------------------

    def _embed_segment_bert(self, text_segment: str) -> np.ndarray:
        inputs = self.tokenizer(
            text_segment,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy().flatten()

    def _embed_segment_openai(self, text_segment: str) -> np.ndarray:
        """Embed a segment using OpenAI embeddings (new client syntax)."""
        response = self._oa_client.embeddings.create(
            model=self.openai_model,
            input=text_segment,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _embed_segment(self, text_segment: str) -> np.ndarray:
        if self.backend == "bert":
            return self._embed_segment_bert(text_segment)
        else:
            return self._embed_segment_openai(text_segment)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        If self.embed_full is False, keeps legacy behaviour (truncate).
        If True, splits the text token sequence into consecutive 512-token
        windows and averages the resulting segment embeddings. This gives a
        single vector representing the *entire* text without truncation.
        """

        if self.backend == "bert" and not self.embed_full:
            return self._embed_segment(text)

        if self.backend == "openai":
            # openai backend context window larger. attempt single call first.
            token_len = len(self.encoding.encode(text))
            if token_len <= self.openai_max_tokens:
                return self._embed_segment_openai(text)

            # otherwise, chunk into windows
            window = self.openai_max_tokens
            stride = self.openai_max_tokens
            tokens = self.encoding.encode(text)
            embeddings = []
            for start in range(0, len(tokens), stride):
                end = start + window
                token_segment = tokens[start:end]
                segment_text = self.encoding.decode(token_segment)
                embeddings.append(self._embed_segment_openai(segment_text))

            aggregate_embedding = np.mean(np.vstack(embeddings), axis=0)
            return aggregate_embedding

        # --- full-text embedding for bert backend ---
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) <= 512:
            return self._embed_segment(text)

        window = 512
        stride = 512
        embeddings = []
        for start in range(0, len(tokens), stride):
            end = start + window
            token_segment = tokens[start:end]
            segment_text = self.tokenizer.decode(
                token_segment,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            embeddings.append(self._embed_segment(segment_text))

        aggregate_embedding = np.mean(np.vstack(embeddings), axis=0)
        return aggregate_embedding

    def create_word_variants(
        self,
        original_text: str,
        num_variants: int = 100,
        words_changed: int = 1,
    ) -> List[Dict]:
        """Create variants by changing *words_changed* words per variant.

        Args:
            original_text: the base text
            num_variants: how many variants to create
            words_changed: number of words to replace in each variant
        """
        words = original_text.split()
        print(f"[debug] original text has {len(words)} words")

        if len(words) < 10:
            print("[error] text too short for meaningful collision testing")
            return []

        variants = []

        # common replacement words for different types
        replacements = {
            "common": [
                "the",
                "and",
                "or",
                "but",
                "with",
                "for",
                "to",
                "of",
                "in",
                "on",
            ],
            "verbs": [
                "is",
                "was",
                "are",
                "were",
                "has",
                "have",
                "had",
                "will",
                "would",
                "could",
            ],
            "nouns": [
                "thing",
                "person",
                "place",
                "time",
                "way",
                "day",
                "man",
                "world",
                "life",
                "hand",
            ],
            "adjectives": [
                "good",
                "new",
                "first",
                "last",
                "long",
                "great",
                "little",
                "own",
                "other",
                "old",
            ],
        }

        all_replacements = []
        for word_list in replacements.values():
            all_replacements.extend(word_list)

        # determine positions to base variants on (even if no words will change)
        if len(words) > num_variants:
            positions_to_test = sorted(random.sample(range(len(words)), num_variants))
        else:
            positions_to_test = list(range(len(words)))[:num_variants]

        for i, pos in enumerate(positions_to_test):
            # when no words are to be changed, keep text identical to original
            if words_changed == 0:
                variants.append(
                    {
                        "variant_id": i,
                        "positions": [],
                        "num_words_changed": 0,
                        "changes": [],
                        "text": original_text,  # exact duplicate
                    }
                )
                if i % 20 == 0:
                    print(
                        f"[debug] created {i+1}/{len(positions_to_test)} variants (0 words changed)"
                    )
                continue

            variant_words = words.copy()
            changed_pairs = []

            # choose *words_changed* unique positions (ensure 'pos' included)
            change_positions = {pos}
            additional_needed = words_changed - 1
            if additional_needed > 0:
                remaining_pool = list(set(range(len(words))) - change_positions)
                change_positions.update(
                    random.sample(
                        remaining_pool, min(additional_needed, len(remaining_pool))
                    )
                )

            for p in change_positions:
                original_word = variant_words[p]
                replacement_word = random.choice(all_replacements)
                variant_words[p] = replacement_word
                changed_pairs.append(
                    {
                        "position": p,
                        "original_word": original_word,
                        "replacement_word": replacement_word,
                    }
                )

            variant_text = " ".join(variant_words)

            variants.append(
                {
                    "variant_id": i,
                    "positions": sorted(list(change_positions)),
                    "num_words_changed": words_changed,
                    "changes": changed_pairs,
                    "text": variant_text,
                }
            )

            if i % 20 == 0:
                print(f"[debug] created {i+1}/{len(positions_to_test)} variants")

        return variants

    def test_collisions(self, original_text: str, variants: List[Dict]) -> Dict:
        """Test for embedding collisions between original and variants."""
        print(f"[debug] testing collisions for {len(variants)} variants")

        # get original embedding
        print("[debug] computing original embedding...")
        original_embedding = self.get_embedding(original_text)

        collision_results = []
        collisions_found = 0
        near_collisions_found = 0

        for i, variant in enumerate(variants):
            variant_embedding = self.get_embedding(variant["text"])

            # compute distance metrics
            euclidean_distance = np.linalg.norm(original_embedding - variant_embedding)
            cosine_sim = cosine_similarity([original_embedding], [variant_embedding])[
                0
            ][0]
            cosine_distance = 1 - cosine_sim

            # check for collisions
            is_collision = euclidean_distance < self.collision_threshold
            is_near_collision = euclidean_distance < self.near_collision_threshold

            if is_collision:
                collisions_found += 1
                if (
                    variant.get("num_words_changed", 1) > 0
                    and "changes" in variant
                    and variant["changes"]
                ):
                    change_desc = ", ".join(
                        [
                            f"{c['original_word']}→{c['replacement_word']}"
                            for c in variant["changes"]
                        ]
                    )
                    print(
                        f"[pass] collision found! variant {i}: {change_desc} (distance: {euclidean_distance:.2e})"
                    )
                else:
                    print(f"[pass] collision found! variant {i} (0-word duplicate)")
            elif is_near_collision:
                near_collisions_found += 1
                print(
                    f"[pass] near collision! variant {i}: distance {euclidean_distance:.2e}"
                )

            collision_results.append(
                {
                    "variant_id": variant["variant_id"],
                    "positions_changed": variant.get(
                        "positions", [variant.get("position", None)]
                    ),
                    "num_words_changed": variant.get("num_words_changed", 1),
                    "changes": variant.get("changes"),
                    "euclidean_distance": euclidean_distance,
                    "cosine_similarity": cosine_sim,
                    "cosine_distance": cosine_distance,
                    "is_collision": is_collision,
                    "is_near_collision": is_near_collision,
                }
            )

            if i % 25 == 0:
                print(f"[debug] processed {i+1}/{len(variants)} variants")

        return {
            "original_text_length": len(original_text.split()),
            "num_variants_tested": len(variants),
            "collisions_found": collisions_found,
            "near_collisions_found": near_collisions_found,
            "collision_rate": collisions_found / len(variants),
            "near_collision_rate": near_collisions_found / len(variants),
            "results": collision_results,
            "collision_threshold": self.collision_threshold,
            "near_collision_threshold": self.near_collision_threshold,
        }

    def analyze_collision_patterns(self, results: Dict) -> Dict:
        """Analyze patterns in collision results."""
        df = pd.DataFrame(results["results"])

        analysis = {
            "distance_stats": {
                "mean_euclidean": df["euclidean_distance"].mean(),
                "std_euclidean": df["euclidean_distance"].std(),
                "min_euclidean": df["euclidean_distance"].min(),
                "max_euclidean": df["euclidean_distance"].max(),
                "mean_cosine_sim": df["cosine_similarity"].mean(),
                "std_cosine_sim": df["cosine_similarity"].std(),
                "min_cosine_sim": df["cosine_similarity"].min(),
                "max_cosine_sim": df["cosine_similarity"].max(),
            },
            "collision_positions": (
                df[df["is_collision"]]["positions_changed"].tolist()
                if results["collisions_found"] > 0
                else []
            ),
            "near_collision_positions": (
                df[df["is_near_collision"]]["positions_changed"].tolist()
                if results["near_collisions_found"] > 0
                else []
            ),
            "closest_variants": df.nsmallest(5, "euclidean_distance")[
                [
                    "euclidean_distance",
                    "cosine_similarity",
                    "num_words_changed",
                    "positions_changed",
                ]
            ].to_dict("records"),
        }

        return analysis

    def visualize_results(self, results: Dict, analysis: Dict, save_path: str = None):
        """Create visualizations of collision test results."""
        df = pd.DataFrame(results["results"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # distance distribution
        axes[0, 0].hist(df["euclidean_distance"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(
            self.collision_threshold,
            color="red",
            linestyle="--",
            label=f"collision threshold ({self.collision_threshold:.0e})",
        )
        axes[0, 0].axvline(
            self.near_collision_threshold,
            color="orange",
            linestyle="--",
            label=f"near collision threshold ({self.near_collision_threshold:.0e})",
        )
        axes[0, 0].set_xlabel("euclidean distance")
        axes[0, 0].set_ylabel("frequency")
        axes[0, 0].set_title("distribution of embedding distances")
        axes[0, 0].legend()
        axes[0, 0].set_yscale("log")

        # distance vs position (robust to empty lists)
        pos_values = df["positions_changed"].apply(
            lambda v: (
                v[0]
                if isinstance(v, list) and len(v) > 0
                else (
                    np.nan if isinstance(v, list) else (v if v is not None else np.nan)
                )
            )
        )
        axes[0, 1].scatter(pos_values, df["euclidean_distance"], alpha=0.6)
        axes[0, 1].axhline(
            self.collision_threshold, color="red", linestyle="--", alpha=0.7
        )
        axes[0, 1].axhline(
            self.near_collision_threshold, color="orange", linestyle="--", alpha=0.7
        )
        axes[0, 1].set_xlabel("word position in text")
        axes[0, 1].set_ylabel("euclidean distance")
        axes[0, 1].set_title("distance vs word position")

        # cosine similarity distribution
        axes[1, 0].hist(df["cosine_similarity"], bins=50, alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("cosine similarity")
        axes[1, 0].set_ylabel("frequency")
        axes[1, 0].set_title("cosine similarity distribution")

        # summary stats
        axes[1, 1].axis("off")
        summary_text = f"""
        collision test summary
        >>> total variants tested: {results['num_variants_tested']}
        >>> exact collisions found: {results['collisions_found']} ({results['collision_rate']:.4f})
        >>> near collisions found: {results['near_collisions_found']} ({results['near_collision_rate']:.4f})
        >>> mean distance: {analysis['distance_stats']['mean_euclidean']:.6f}
        >>> min distance: {analysis['distance_stats']['min_euclidean']:.6f}
        >>> mean cosine similarity: {analysis['distance_stats']['mean_cosine_sim']:.6f}
        """
        axes[1, 1].text(
            0.1,
            0.5,
            summary_text,
            fontsize=12,
            verticalalignment="center",
            fontfamily="monospace",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[debug] visualization saved to: {save_path}")

        plt.show()

    def save_results(self, results: Dict, analysis: Dict, output_dir: str):
        """Save collision test results."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # derive run parameters for filename
        num_variants = results.get("num_variants_tested", len(results["results"]))
        # assume all variants have same num_words_changed; fall back to 'x'
        try:
            words_changed = results["results"][0]["num_words_changed"]
        except (KeyError, IndexError):
            words_changed = "x"

        # save detailed results
        df = pd.DataFrame(results["results"])
        df_filename = (
            f"collision_results_{words_changed}words_{num_variants}var_{timestamp}.csv"
        )
        df.to_csv(os.path.join(output_dir, df_filename), index=False)

        # save summary
        summary_filename = (
            f"collision_summary_{words_changed}words_{num_variants}var_{timestamp}.txt"
        )
        with open(os.path.join(output_dir, summary_filename), "w") as f:
            f.write("embedding collision test results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"original text length: {results['original_text_length']} words\n")
            f.write(f"variants tested: {results['num_variants_tested']}\n")
            f.write(f"collision threshold: {results['collision_threshold']:.2e}\n")
            f.write(
                f"near collision threshold: {results['near_collision_threshold']:.2e}\n\n"
            )
            f.write(
                f"exact collisions found: {results['collisions_found']} ({results['collision_rate']:.6f})\n"
            )
            f.write(
                f"near collisions found: {results['near_collisions_found']} ({results['near_collision_rate']:.6f})\n\n"
            )
            f.write("distance statistics:\n")
            f.write(
                f"  mean euclidean distance: {analysis['distance_stats']['mean_euclidean']:.6f}\n"
            )
            f.write(
                f"  std euclidean distance: {analysis['distance_stats']['std_euclidean']:.6f}\n"
            )
            f.write(
                f"  min euclidean distance: {analysis['distance_stats']['min_euclidean']:.6f}\n"
            )
            f.write(
                f"  max euclidean distance: {analysis['distance_stats']['max_euclidean']:.6f}\n\n"
            )
            f.write("closest variants (top 5):\n")
            for i, variant in enumerate(analysis["closest_variants"]):
                f.write(
                    f"  {i+1}. {variant['num_words_changed']} words changed: {variant['positions_changed']}\n"
                )
                f.write(
                    f"    euclidean_distance: {variant['euclidean_distance']:.6f}    cosine_similarity: {variant['cosine_similarity']:.6f}\n"
                )

        print(f"[debug] results saved to: {output_dir} (timestamp {timestamp})")


def main():
    """Run the embedding collision test."""
    parser = argparse.ArgumentParser(description="test embedding collisions")
    parser.add_argument("--pdf", type=str, default="paper.pdf", help="path to pdf file")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="collision_results",
        help="folder to store results",
    )
    parser.add_argument(
        "--variants", type=int, default=100, help="number of variants to test"
    )
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="bert model to use"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=0,
        help="optionally limit words processed (0 = no limit)",
    )
    parser.add_argument(
        "--words-changed",
        type=int,
        default=1,
        help="how many words to change per variant",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="bert",
        choices=["bert", "openai"],
        help="embedding backend to use",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-embedding-3-small",
        help="openai embedding model name",
    )

    args = parser.parse_args()

    print("embedding collision test")
    print("=" * 40)
    print(f"pdf: {args.pdf}")
    print(f"variants to test: {args.variants}")

    effective_model = args.openai_model if args.backend == "openai" else args.model
    print(f"backend: {args.backend}  model: {effective_model}")
    print("=" * 40)

    # initialize tester
    tester = EmbeddingCollisionTester(
        model_name=args.openai_model if args.backend == "openai" else args.model,
        embed_full=True,
        backend=args.backend,
    )

    # extract text
    print("\n[debug] extracting text from pdf...")
    text = tester.extract_text_from_pdf(args.pdf)
    if not text:
        print("[error] no text extracted!")
        return

    # canonicalize whitespace so baseline & variants share same serialization
    canonical_text = " ".join(text.split())
    if canonical_text != text:
        print("[debug] canonicalized whitespace – baseline will use collapsed spaces")
    text = canonical_text

    print(f"[debug] using text with {len(text.split())} words (canonical)")

    # optional truncation only if user specified positive limit
    if args.max_text_length and args.max_text_length > 0:
        words = text.split()
        if len(words) > args.max_text_length:
            text = " ".join(words[: args.max_text_length])
            print(f"[debug] truncated to {args.max_text_length} words (user limit)")

    print(f"[debug] using text with {len(text.split())} words")

    # create variants
    print("\n[debug] creating text variants...")
    variants = tester.create_word_variants(
        text, num_variants=args.variants, words_changed=args.words_changed
    )
    if not variants:
        return

    # test for collisions
    print("\n[debug] testing for embedding collisions...")
    results = tester.test_collisions(text, variants)

    # analyze results
    print("\n[debug] analyzing collision patterns...")
    analysis = tester.analyze_collision_patterns(results)

    # (visualization step removed)

    # Save all results
    print("\n6. Saving results...")
    tester.save_results(results, analysis, args.output_dir)

    # print summary
    print("\n" + "=" * 40)
    print("collision test completed!")
    print(
        f">>> exact collisions: {results['collisions_found']}/{results['num_variants_tested']} ({results['collision_rate']:.6f})"
    )
    print(
        f">>> near collisions: {results['near_collisions_found']}/{results['num_variants_tested']} ({results['near_collision_rate']:.6f})"
    )
    print(f">>> mean distance: {analysis['distance_stats']['mean_euclidean']:.6f}")
    print(f">>> results saved to: {args.output_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()

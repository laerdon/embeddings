#!/usr/bin/env python3
"""
Quick Start Example for BERT Embeddings Experiment

This script demonstrates basic usage of the BERT embedding functionality
with your PDF document.

Run this script to get started quickly with default settings.
"""

import os
import sys
from bert_embeddings_experiment import BERTEmbeddingExtractor

def quick_demo():
    """Run a quick demonstration with the available PDF."""
    
    print("BERT Embeddings Quick Start Demo")
    print("=" * 40)
    
    # Check if PDF exists
    pdf_path = "paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        print("Please ensure you have a PDF file named 'paper.pdf' in the current directory.")
        return
    
    try:
        # Initialize BERT extractor
        print("Initializing BERT model...")
        extractor = BERTEmbeddingExtractor()
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        text = extractor.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print("Error: No text could be extracted from the PDF!")
            return
        
        print(f"✓ Extracted {len(text)} characters from PDF")
        
        # Create text chunks
        print("Creating text chunks...")
        chunks = extractor.chunk_text(text, chunk_size=300, overlap=30)
        print(f"✓ Created {len(chunks)} text chunks")
        
        # Limit to first 5 chunks for quick demo
        demo_chunks = chunks[:5]
        print(f"Using first {len(demo_chunks)} chunks for quick demo")
        
        # Generate embeddings
        print("Generating BERT embeddings...")
        embeddings = extractor.generate_embeddings_batch(demo_chunks, batch_size=2)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        
        # Analyze similarity
        print("Analyzing similarity between chunks...")
        similarity_analysis = extractor.analyze_similarity(embeddings, demo_chunks, top_k=3)
        
        print(f"\nSimilarity Analysis Results:")
        print(f"Mean similarity: {similarity_analysis['mean_similarity']:.4f}")
        print(f"Standard deviation: {similarity_analysis['std_similarity']:.4f}")
        
        print(f"\nTop 3 most similar chunk pairs:")
        for i, pair in enumerate(similarity_analysis['top_similar_pairs'][:3]):
            print(f"\nPair {i+1} (Similarity: {pair['similarity']:.4f}):")
            print(f"Chunk A: {pair['text1_preview']}")
            print(f"Chunk B: {pair['text2_preview']}")
        
        # Quick clustering (if we have enough chunks)
        if len(demo_chunks) >= 3:
            print(f"\nClustering {len(demo_chunks)} chunks into 3 groups...")
            cluster_labels, silhouette_score = extractor.cluster_embeddings(embeddings, n_clusters=3)
            print(f"Silhouette score: {silhouette_score:.4f}")
            
            print(f"\nCluster assignments:")
            for i, (chunk, label) in enumerate(zip(demo_chunks, cluster_labels)):
                preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
                print(f"Chunk {i+1} → Cluster {label}: {preview}")
        
        print(f"\n" + "=" * 40)
        print("Quick demo completed successfully!")
        print("To run the full experiment with all chunks, use:")
        print("python bert_embeddings_experiment.py --pdf paper.pdf")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure your virtual environment is activated and all packages are installed.")

if __name__ == "__main__":
    quick_demo()
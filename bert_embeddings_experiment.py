#!/usr/bin/env python3
"""
BERT Embeddings Experiment Script

This script provides functionality to:
1. Extract text from PDF files
2. Generate BERT embeddings for text chunks
3. Perform similarity analysis
4. Visualize embedding clusters
5. Save and load embeddings for future use

Usage:
    python bert_embeddings_experiment.py --pdf paper.pdf --output embeddings_results
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import PyPDF2
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class BERTEmbeddingExtractor:
    """Class to handle BERT embedding extraction and analysis."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize the BERT model and tokenizer.
        
        Args:
            model_name: HuggingFace model name (default: bert-base-uncased)
        """
        print(f"Loading BERT model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum number of words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Generate BERT embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            BERT embedding as numpy array
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().flatten()
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate BERT embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                embedding = self.get_bert_embedding(text)
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def analyze_similarity(self, embeddings: np.ndarray, texts: List[str], top_k: int = 5) -> Dict:
        """
        Analyze similarity between text chunks.
        
        Args:
            embeddings: Array of embeddings
            texts: Corresponding texts
            top_k: Number of most similar pairs to return
            
        Returns:
            Dictionary with similarity analysis results
        """
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find most similar pairs (excluding self-similarity)
        similarity_pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity_pairs.append({
                    'text1_idx': i,
                    'text2_idx': j,
                    'similarity': similarity_matrix[i, j],
                    'text1_preview': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                    'text2_preview': texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j]
                })
        
        # Sort by similarity and return top-k
        similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'similarity_matrix': similarity_matrix,
            'top_similar_pairs': similarity_pairs[:top_k],
            'mean_similarity': np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
            'std_similarity': np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        }
    
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, float]:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels, silhouette score)
        """
        from sklearn.metrics import silhouette_score
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        try:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        except:
            silhouette_avg = 0.0
        
        return cluster_labels, silhouette_avg
    
    def visualize_embeddings(self, embeddings: np.ndarray, labels: np.ndarray = None, 
                           title: str = "BERT Embeddings Visualization", save_path: str = None):
        """
        Visualize embeddings using PCA dimensionality reduction.
        
        Args:
            embeddings: Array of embeddings
            labels: Optional cluster labels for coloring
            title: Plot title
            save_path: Path to save the plot
        """
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=labels, cmap='tab10', alpha=0.7, s=50)
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
        
        plt.title(f'{title}\nExplained Variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}')
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save experiment results to files.
        
        Args:
            results: Dictionary containing all results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings
        np.save(os.path.join(output_dir, 'embeddings.npy'), results['embeddings'])
        
        # Save text chunks
        with open(os.path.join(output_dir, 'text_chunks.txt'), 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(results['text_chunks']):
                f.write(f"--- Chunk {i + 1} ---\n")
                f.write(chunk)
                f.write("\n\n")
        
        # Save similarity analysis
        with open(os.path.join(output_dir, 'similarity_analysis.txt'), 'w', encoding='utf-8') as f:
            sim_analysis = results['similarity_analysis']
            f.write("SIMILARITY ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Mean similarity: {sim_analysis['mean_similarity']:.4f}\n")
            f.write(f"Std similarity: {sim_analysis['std_similarity']:.4f}\n\n")
            f.write("TOP SIMILAR PAIRS:\n")
            f.write("-" * 30 + "\n")
            
            for i, pair in enumerate(sim_analysis['top_similar_pairs']):
                f.write(f"\nPair {i + 1} (Similarity: {pair['similarity']:.4f}):\n")
                f.write(f"Text 1: {pair['text1_preview']}\n")
                f.write(f"Text 2: {pair['text2_preview']}\n")
                f.write("-" * 30 + "\n")
        
        # Save cluster analysis
        if 'cluster_labels' in results:
            cluster_df = pd.DataFrame({
                'chunk_id': range(len(results['text_chunks'])),
                'cluster': results['cluster_labels'],
                'text_preview': [chunk[:100] + "..." if len(chunk) > 100 else chunk 
                               for chunk in results['text_chunks']]
            })
            cluster_df.to_csv(os.path.join(output_dir, 'clusters.csv'), index=False)
        
        # Save metadata
        metadata = {
            'num_chunks': len(results['text_chunks']),
            'embedding_dimension': results['embeddings'].shape[1],
            'model_used': 'bert-base-uncased',
            'similarity_stats': {
                'mean': float(results['similarity_analysis']['mean_similarity']),
                'std': float(results['similarity_analysis']['std_similarity'])
            }
        }
        
        if 'silhouette_score' in results:
            metadata['clustering'] = {
                'n_clusters': len(np.unique(results['cluster_labels'])),
                'silhouette_score': float(results['silhouette_score'])
            }
        
        with open(os.path.join(output_dir, 'metadata.pickle'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Results saved to: {output_dir}")

def main():
    """Main function to run the BERT embedding experiment."""
    parser = argparse.ArgumentParser(description='BERT Embeddings Experiment')
    parser.add_argument('--pdf', type=str, default='paper.pdf', 
                       help='Path to PDF file to analyze')
    parser.add_argument('--output', type=str, default='embeddings_results',
                       help='Output directory for results')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Size of text chunks (words)')
    parser.add_argument('--overlap', type=int, default=50,
                       help='Overlap between chunks (words)')
    parser.add_argument('--clusters', type=int, default=5,
                       help='Number of clusters for K-means')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='BERT model to use')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("BERT Embeddings Experiment")
    print("=" * 50)
    print(f"PDF file: {args.pdf}")
    print(f"Output directory: {args.output}")
    print(f"Chunk size: {args.chunk_size} words")
    print(f"Overlap: {args.overlap} words")
    print(f"Number of clusters: {args.clusters}")
    print(f"Model: {args.model}")
    print("=" * 50)
    
    # Initialize the BERT extractor
    extractor = BERTEmbeddingExtractor(model_name=args.model)
    
    # Extract text from PDF
    print("\n1. Extracting text from PDF...")
    text = extractor.extract_text_from_pdf(args.pdf)
    if not text.strip():
        print("Error: No text extracted from PDF!")
        return
    
    print(f"Extracted {len(text)} characters")
    
    # Chunk the text
    print("\n2. Chunking text...")
    chunks = extractor.chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} text chunks")
    
    # Generate embeddings
    print("\n3. Generating BERT embeddings...")
    embeddings = extractor.generate_embeddings_batch(chunks, batch_size=args.batch_size)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Analyze similarity
    print("\n4. Analyzing similarity...")
    similarity_analysis = extractor.analyze_similarity(embeddings, chunks)
    print(f"Mean similarity: {similarity_analysis['mean_similarity']:.4f}")
    print(f"Std similarity: {similarity_analysis['std_similarity']:.4f}")
    
    # Cluster embeddings
    print("\n5. Clustering embeddings...")
    cluster_labels, silhouette_score = extractor.cluster_embeddings(embeddings, n_clusters=args.clusters)
    print(f"Silhouette score: {silhouette_score:.4f}")
    
    # Visualize embeddings
    print("\n6. Creating visualizations...")
    vis_path = os.path.join(args.output, 'embeddings_visualization.png')
    extractor.visualize_embeddings(embeddings, cluster_labels, 
                                 title="BERT Embeddings Clustering", 
                                 save_path=vis_path)
    
    # Save all results
    print("\n7. Saving results...")
    results = {
        'embeddings': embeddings,
        'text_chunks': chunks,
        'similarity_analysis': similarity_analysis,
        'cluster_labels': cluster_labels,
        'silhouette_score': silhouette_score
    }
    
    extractor.save_results(results, args.output)
    
    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output}")
    print("=" * 50)

if __name__ == "__main__":
    main()
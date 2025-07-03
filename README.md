# BERT Embeddings Experiment

A comprehensive environment for extracting and analyzing BERT embeddings from PDF documents. This project provides tools for text extraction, embedding generation, similarity analysis, clustering, and visualization.

## 🚀 Quick Start

### 1. Activate the Virtual Environment
```bash
source bert_experiment_env/bin/activate
```

### 2. Run Quick Demo
```bash
python quick_start.py
```

### 3. Full Experiment
```bash
python bert_embeddings_experiment.py --pdf paper.pdf --output results
```

## 📋 Features

- **PDF Text Extraction**: Extract text content from PDF documents
- **BERT Embeddings**: Generate high-quality embeddings using pre-trained BERT models
- **Similarity Analysis**: Find the most similar text chunks using cosine similarity
- **Clustering**: Group similar content using K-means clustering
- **Visualization**: Create 2D visualizations of embeddings using PCA
- **Export Results**: Save embeddings, analysis, and visualizations for further use

## 🛠 Installation

The environment is already set up with all required packages:

- **PyTorch 2.7.1** with CUDA 12.6 support
- **Transformers 4.53.0** for BERT models
- **PyPDF2 3.0.1** for PDF text extraction
- **scikit-learn 1.7.0** for machine learning tools
- **matplotlib 3.10.3** and **seaborn 0.13.2** for visualization
- **numpy 2.3.1** and **pandas 2.3.0** for data processing

## 📖 Usage Examples

### Basic Usage
```bash
# Process a PDF with default settings
python bert_embeddings_experiment.py --pdf paper.pdf

# Specify output directory
python bert_embeddings_experiment.py --pdf paper.pdf --output my_results

# Customize text chunking
python bert_embeddings_experiment.py --pdf paper.pdf --chunk-size 1000 --overlap 100

# Use different number of clusters
python bert_embeddings_experiment.py --pdf paper.pdf --clusters 10

# Use a different BERT model
python bert_embeddings_experiment.py --pdf paper.pdf --model bert-large-uncased
```

### Command Line Options
```
--pdf             Path to PDF file (default: paper.pdf)
--output          Output directory (default: embeddings_results)
--chunk-size      Text chunk size in words (default: 500)
--overlap         Overlap between chunks in words (default: 50)
--clusters        Number of clusters for K-means (default: 5)
--model           BERT model to use (default: bert-base-uncased)
--batch-size      Processing batch size (default: 8)
```

### Python API Usage
```python
from bert_embeddings_experiment import BERTEmbeddingExtractor

# Initialize the extractor
extractor = BERTEmbeddingExtractor(model_name='bert-base-uncased')

# Extract text from PDF
text = extractor.extract_text_from_pdf('paper.pdf')

# Create text chunks
chunks = extractor.chunk_text(text, chunk_size=500, overlap=50)

# Generate embeddings
embeddings = extractor.generate_embeddings_batch(chunks)

# Analyze similarity
similarity_results = extractor.analyze_similarity(embeddings, chunks)

# Cluster embeddings
cluster_labels, silhouette_score = extractor.cluster_embeddings(embeddings, n_clusters=5)

# Visualize results
extractor.visualize_embeddings(embeddings, cluster_labels)
```

## 📊 Output Files

The experiment generates several output files:

- **`embeddings.npy`**: NumPy array of BERT embeddings
- **`text_chunks.txt`**: Original text chunks used for analysis
- **`similarity_analysis.txt`**: Detailed similarity analysis results
- **`clusters.csv`**: Cluster assignments for each text chunk
- **`embeddings_visualization.png`**: 2D PCA visualization of embeddings
- **`metadata.pickle`**: Experiment metadata and statistics

## 🔧 Advanced Configuration

### Using Different Models
The script supports any BERT-compatible model from Hugging Face:
```bash
# Use BERT Large
python bert_embeddings_experiment.py --model bert-large-uncased

# Use RoBERTa
python bert_embeddings_experiment.py --model roberta-base

# Use DistilBERT (faster)
python bert_embeddings_experiment.py --model distilbert-base-uncased
```

### Memory and Performance
- **GPU**: Automatically uses CUDA if available
- **Batch Size**: Adjust `--batch-size` based on your GPU memory
- **Chunk Size**: Larger chunks provide more context but use more memory

## 📈 Understanding Results

### Similarity Analysis
- **Mean Similarity**: Average cosine similarity between all chunk pairs
- **Standard Deviation**: Variation in similarity scores
- **Top Similar Pairs**: Most semantically similar text chunks

### Clustering
- **Silhouette Score**: Quality of clustering (higher is better, range: -1 to 1)
- **Cluster Labels**: Group assignment for each text chunk

### Visualization
- **PCA Plot**: 2D representation of high-dimensional embeddings
- **Explained Variance**: How much information is preserved in 2D

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python bert_embeddings_experiment.py --batch-size 4
   ```

2. **PDF Reading Errors**:
   - Ensure PDF is not password-protected
   - Try with a different PDF file

3. **Model Download Issues**:
   - Check internet connection
   - Models are downloaded automatically on first use

### System Requirements
- **RAM**: 8GB+ recommended
- **GPU**: Optional but recommended for faster processing
- **Disk Space**: 2-4GB for models and results

## 📚 Examples and Use Cases

- **Research Paper Analysis**: Find similar sections in academic papers
- **Document Clustering**: Group related documents or paragraphs
- **Content Similarity**: Identify duplicate or similar content
- **Information Retrieval**: Build semantic search systems
- **Text Mining**: Extract patterns from large document collections

## 🤝 Contributing

Feel free to extend the functionality by:
- Adding support for other document formats
- Implementing additional clustering algorithms
- Creating more visualization options
- Adding evaluation metrics

#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that all required packages are properly installed
and the BERT embeddings environment is ready for use.
"""

import sys
import importlib
import torch

def check_package(package_name, optional=False):
    """Check if a package is installed and importable."""
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        status = "⚠️" if optional else "✗"
        print(f"{status} {package_name} - {'Optional' if optional else 'MISSING'}")
        return not optional

def check_torch_cuda():
    """Check PyTorch and CUDA availability."""
    print(f"✓ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
        print(f"  - Device 0: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    else:
        print("⚠️ CUDA not available (will use CPU)")
    
    return True

def test_bert_model():
    """Test loading a simple BERT model."""
    print("\nTesting BERT model loading...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Load a small model for testing
        model_name = "distilbert-base-uncased"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test tokenization and inference
        test_text = "This is a test sentence for BERT."
        inputs = tokenizer(test_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Test inference completed")
        print(f"  - Input shape: {inputs['input_ids'].shape}")
        print(f"  - Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing BERT model: {e}")
        return False

def test_pdf_reading():
    """Test PDF reading capability."""
    print("\nTesting PDF reading...")
    try:
        import PyPDF2
        
        # Check if sample PDF exists
        import os
        if os.path.exists("paper.pdf"):
            print("✓ Sample PDF found: paper.pdf")
            
            # Try to read the first page
            with open("paper.pdf", 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"✓ PDF has {num_pages} pages")
                
                if num_pages > 0:
                    first_page_text = pdf_reader.pages[0].extract_text()
                    print(f"✓ First page text length: {len(first_page_text)} characters")
        else:
            print("⚠️ No sample PDF found (paper.pdf)")
            print("  You can add a PDF file to test the complete pipeline")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing PDF reading: {e}")
        return False

def main():
    """Run all verification checks."""
    print("BERT Embeddings Environment Verification")
    print("=" * 50)
    
    print("\n1. Checking required packages...")
    packages_ok = True
    
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn',
        'PyPDF2'
    ]
    
    for package in required_packages:
        if not check_package(package):
            packages_ok = False
    
    print("\n2. Checking PyTorch and CUDA...")
    torch_ok = check_torch_cuda()
    
    print("\n3. Testing BERT model loading...")
    bert_ok = test_bert_model()
    
    print("\n4. Testing PDF reading...")
    pdf_ok = test_pdf_reading()
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    if packages_ok and torch_ok and bert_ok:
        print("🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Run quick demo: python quick_start.py")
        print("2. Run full experiment: python bert_embeddings_experiment.py")
    else:
        print("⚠️ Some issues detected. Please check the errors above.")
        
        if not packages_ok:
            print("\nTo fix package issues:")
            print("source bert_experiment_env/bin/activate")
            print("pip install torch transformers PyPDF2 numpy scikit-learn matplotlib seaborn pandas")
    
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    return packages_ok and torch_ok and bert_ok

if __name__ == "__main__":
    main()
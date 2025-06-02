#!/usr/bin/env python3
"""
Test script for Llama attention analysis.

This script tests the updated attention visualization functionality
specifically for standard Llama models.
"""

import sys
import os
import torch

# Add src to path
sys.path.append('src')

def test_llama_attention():
    """Test Llama attention extraction and visualization."""
    try:
        from src.attention_visualizer import visualize_sample_attention, get_attention_scores
        from src.model_loader import load_model_and_tokenizer
        
        print("="*80)
        print("TESTING LLAMA ATTENTION ANALYSIS")
        print("="*80)
        
        # Test question
        test_question = "What is the capital of France?"
        
        print(f"Test question: {test_question}")
        print("Loading Llama model...")
        
        # Test model loading
        device = "cpu"  # Use CPU for testing
        model, tokenizer = load_model_and_tokenizer("llama", device)
        
        print(f"Model loaded: {type(model).__name__}")
        print(f"Config: {model.config.model_type}")
        
        # Test attention extraction
        print("\nTesting attention extraction...")
        
        attention_matrix, tokens, metadata = get_attention_scores(
            model, tokenizer, test_question, device, "llama", layer_idx=-1, head_idx=0
        )
        
        print(f"Results:")
        print(f"  Attention matrix: {attention_matrix.shape if attention_matrix is not None else 'None'}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Metadata keys: {list(metadata.keys())}")
        
        assert attention_matrix is not None, "Attention matrix should be extracted for Llama"
        assert 'lambda_params' not in metadata, "Llama metadata should NOT contain lambda_params"
        assert 'lambda_std_dev' not in metadata, "Llama metadata should NOT contain lambda_std_dev"
        assert 'is_diffllama' not in metadata or not metadata['is_diffllama'], "is_diffllama should be False or absent for Llama"

        # Test visualization
        print("\nTesting visualization...")
        
        os.makedirs("test_results", exist_ok=True)
        
        visualize_sample_attention(
            "llama", 
            test_question, 
            layer_idx=-1, 
            head_idx=0, 
            save_dir="test_results/llama_test"
        )
        
        print("✅ Llama attention analysis test completed successfully!")
        
        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Llama Attention Analysis Test Suite")
    print("=" * 60)
    
    success = test_llama_attention()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Llama attention test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 Llama attention analysis test passed!")
        print("Check test_results/llama_test directory for generated visualizations.")
    else:
        print("\n❌ Llama attention analysis test failed. Check error messages above.")
    
    print("\nNext steps:")
    print("1. Run the full attention visualizer: python -m src.attention_visualizer")
    print("2. Check results/attention_maps/ for comprehensive analysis.") 
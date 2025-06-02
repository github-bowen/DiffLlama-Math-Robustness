#!/usr/bin/env python3
"""
Test script for DiffLlama attention analysis.

This script tests the updated attention visualization functionality
specifically for DiffLlama models.
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_diffllama_attention():
    """Test DiffLlama attention extraction and visualization."""
    try:
        from src.attention_visualizer import visualize_sample_attention, get_attention_scores
        from src.model_loader import load_model_and_tokenizer
        
        print("="*80)
        print("TESTING DIFFLAMA ATTENTION ANALYSIS")
        print("="*80)
        
        # Test question
        test_question = "What is 2 + 3?"
        
        print(f"Test question: {test_question}")
        print("Loading DiffLlama model...")
        
        # Test model loading
        device = "cpu"  # Use CPU for testing
        model, tokenizer = load_model_and_tokenizer("diffllama", device)
        
        print(f"Model loaded: {type(model).__name__}")
        print(f"Config: {model.config.model_type}")
        print(f"Lambda std dev (config): {getattr(model.config, 'lambda_std_dev', 'N/A')}")
        
        # Test attention extraction
        print("\nTesting attention extraction...")
        
        attention_matrix, tokens, metadata = get_attention_scores(
            model, tokenizer, test_question, device, "diffllama", layer_idx=-1, head_idx=0
        )
        
        print(f"Results:")
        print(f"  Attention matrix: {attention_matrix.shape if attention_matrix is not None else 'None'}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Metadata (first 5 keys): { {k: metadata[k] for k in list(metadata.keys())[:5]} }...")
        assert attention_matrix is not None, "Attention matrix should be extracted for DiffLlama"
        assert 'lambda_params' in metadata, "DiffLlama metadata should contain lambda_params"
        assert 'lambda_std_dev' in metadata, "DiffLlama metadata should contain lambda_std_dev from config"
        
        if metadata.get('lambda_params'):
            print(f"  Lambda params found: {list(metadata['lambda_params'].keys())}")
        else:
            print("  No specific lambda_params captured from module attributes.")

        # Test visualization
        print("\nTesting visualization...")
        
        os.makedirs("test_results", exist_ok=True)
        
        visualize_sample_attention(
            "diffllama", 
            test_question, 
            layer_idx=-1, 
            head_idx=0, 
            save_dir="test_results/diffllama_test"
        )
        
        print("✅ DiffLlama attention analysis test completed successfully!")
        
        # Clean up
        del model, tokenizer
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison():
    """Test comparison between Llama and DiffLlama."""
    try:
        from src.attention_visualizer import get_attention_scores
        from src.model_loader import load_model_and_tokenizer
        
        print("\n" + "="*80)
        print("TESTING LLAMA VS DIFFLAMA COMPARISON")
        print("="*80)
        
        test_question = "What is 5 + 7?"
        device = "cpu"
        
        results = {}
        
        for model_type in ["llama", "diffllama"]:
            print(f"\nTesting {model_type}...")
            
            model, tokenizer = load_model_and_tokenizer(model_type, device)
            
            attention_matrix, tokens, metadata = get_attention_scores(
                model, tokenizer, test_question, device, model_type, layer_idx=-1, head_idx=0
            )
            
            results[model_type] = {
                'attention_available': attention_matrix is not None,
                'metadata': metadata
            }
            
            print(f"  {model_type} results:")
            print(f"    Attention: {'✅' if attention_matrix is not None else '❌'}")
            if model_type == "diffllama":
                print(f"    Lambda params captured: {'✅' if metadata.get('lambda_params') else '❌'}")
            
            del model, tokenizer
        
        # Compare results
        print(f"\nComparison Summary:")
        llama_meta = results['llama']['metadata']
        diffllama_meta = results['diffllama']['metadata']
        
        print(f"  Llama captured hook components: {len(llama_meta['captured_components'])}")
        print(f"  DiffLlama captured hook components: {len(diffllama_meta['captured_components'])}")
        print(f"  DiffLlama lambda_std_dev (config): {diffllama_meta.get('lambda_std_dev', 'N/A')}")
        
        diffllama_has_lambda_module_params = bool(diffllama_meta.get('lambda_params'))
        print(f"  DiffLlama lambda params from module detected: {'✅' if diffllama_has_lambda_module_params else '❌'}")
        
        assert results['llama']['attention_available'], "Llama should provide an attention matrix."
        assert results['diffllama']['attention_available'], "DiffLlama should provide an attention matrix."
        assert diffllama_has_lambda_module_params or diffllama_meta.get('lambda_std_dev') is not None, "DiffLlama should have some lambda parameters (from module or config)."

        print("✅ Comparison test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DiffLlama Attention Analysis Test Suite")
    print("=" * 60)
    
    # Test 1: Basic DiffLlama attention
    success1 = test_diffllama_attention()
    
    # Test 2: Comparison
    success2 = test_comparison()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"DiffLlama attention test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Comparison test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! DiffLlama attention analysis is working.")
        print("Check test_results/ directory for generated visualizations.")
    else:
        print("\n❌ Some tests failed. Check error messages above.")
    
    print("\nNext steps:")
    print("1. Run the full attention visualizer: python -m src.attention_visualizer")
    print("2. Check results/attention_maps/ for comprehensive analysis.")
    print("3. For DiffLlama, observe the reported lambda parameters in the console output and metadata.") 
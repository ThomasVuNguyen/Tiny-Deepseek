"""
Process a small sample of the FineWeb dataset to test the processor
"""

import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.fineweb_processor import FineWebDataProcessor

def main():
    print("[+] Processing FineWeb dataset sample...")
    
    # Create processor
    processor = FineWebDataProcessor()
    
    # Test with a small sample first
    print("Testing processor with sample data...")
    
    # Test the preprocessing function
    sample_text = "This is a sample educational content about machine learning and artificial intelligence. It contains useful information for students and researchers."
    processed = processor.preprocess_text(sample_text)
    print(f"Sample text processed: {processed[:100]}...")
    
    # Test the content extraction
    sample_example = {
        'text': sample_text,
        'url': 'https://example.com/ai-article',
        'date': '2024-01-15',
        'language': 'en',
        'score': 0.95
    }
    
    elements = processor.extract_content_elements(sample_example)
    print(f"Extracted elements: {elements}")
    
    # Test the processing function
    processed_example = processor.process(sample_example)
    print(f"Processed example tokens: {len(processed_example['ids'])}")
    
    print("[+] FineWeb processor test completed successfully!")

if __name__ == "__main__":
    main()

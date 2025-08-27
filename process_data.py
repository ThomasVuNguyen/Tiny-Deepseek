import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_processor import DeepSeekDataProcessor

def main():
    print("[+] Processing dataset into binary files...")
    processor = DeepSeekDataProcessor()
    processor.prepare_dataset()
    print("[+] Data processing completed successfully!")

if __name__ == "__main__":
    main()

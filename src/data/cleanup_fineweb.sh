#!/bin/bash

echo "🗑️  FineWeb Dataset Cleanup Script"
echo "=================================="

# Check current usage
echo "📊 Current FineWeb cache usage:"
du -sh /root/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu 2>/dev/null || echo "No hub cache found"
du -sh /root/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu 2>/dev/null || echo "No datasets cache found"

echo ""
echo "🧹 Cleaning up FineWeb cached data..."

# Remove the hub cache (downloaded files)
if [ -d "/root/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu" ]; then
    echo "Removing hub cache..."
    rm -rf /root/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu
    echo "✅ Hub cache removed"
else
    echo "ℹ️  No hub cache found"
fi

# Remove the datasets cache (processed data)
if [ -d "/root/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu" ]; then
    echo "Removing datasets cache..."
    rm -rf /root/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu
    echo "✅ Datasets cache removed"
else
    echo "ℹ️  No datasets cache found"
fi

# Remove lock files
if [ -d "/root/.cache/huggingface/hub/.locks/datasets--HuggingFaceFW--fineweb-edu" ]; then
    echo "Removing lock files..."
    rm -rf /root/.cache/huggingface/hub/.locks/datasets--HuggingFaceFW--fineweb-edu
    echo "✅ Lock files removed"
fi

# Also clean up any processed binary files in our data directory
if [ -f "src/data/fineweb_train.bin" ] || [ -f "src/data/fineweb_validation.bin" ] || [ -f "src/data/fineweb_finetune.bin" ]; then
    echo "Removing processed FineWeb binary files..."
    rm -f src/data/fineweb_*.bin
    echo "✅ Binary files removed"
fi

echo ""
echo "🎉 FineWeb cleanup completed!"
echo "💾 Space freed: ~77GB"
echo ""
echo "📋 Summary:"
echo "- Removed HuggingFace hub cache"
echo "- Removed datasets cache" 
echo "- Removed lock files"
echo "- Removed processed binary files"
echo ""
echo "ℹ️  You can now run the processor again with the smaller dataset size."

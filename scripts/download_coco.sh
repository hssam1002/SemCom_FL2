#!/bin/bash
# Script to download COCO 2017 training dataset

DATA_ROOT="/data4/hongsik/data/COCO"
mkdir -p "${DATA_ROOT}/annotations"
mkdir -p "${DATA_ROOT}/train2017"

echo "=========================================="
echo "COCO Dataset Download Script"
echo "=========================================="
echo ""
echo "This script will download:"
echo "1. COCO 2017 Training Images (~18GB)"
echo "2. COCO 2017 Annotations (~250MB)"
echo ""
echo "Target directory: ${DATA_ROOT}"
echo ""

# Download training images
echo "Downloading COCO 2017 Training Images..."
echo "URL: http://images.cocodataset.org/zips/train2017.zip"
echo "This may take a while (~18GB)..."
wget -c http://images.cocodataset.org/zips/train2017.zip -O /tmp/train2017.zip

if [ $? -eq 0 ]; then
    echo "Extracting training images..."
    unzip -q /tmp/train2017.zip -d "${DATA_ROOT}"
    rm /tmp/train2017.zip
    echo "✓ Training images extracted to ${DATA_ROOT}/train2017/"
else
    echo "✗ Failed to download training images"
    exit 1
fi

# Download annotations
echo ""
echo "Downloading COCO 2017 Annotations..."
echo "URL: http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /tmp/annotations_trainval2017.zip

if [ $? -eq 0 ]; then
    echo "Extracting annotations..."
    unzip -q /tmp/annotations_trainval2017.zip -d /tmp
    cp /tmp/annotations/captions_train2017.json "${DATA_ROOT}/annotations/"
    rm -rf /tmp/annotations /tmp/annotations_trainval2017.zip
    echo "✓ Annotations extracted to ${DATA_ROOT}/annotations/captions_train2017.json"
else
    echo "✗ Failed to download annotations"
    exit 1
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Dataset structure:"
echo "${DATA_ROOT}/"
echo "├── annotations/"
echo "│   └── captions_train2017.json"
echo "└── train2017/"
echo "    └── *.jpg (118,287 images)"
echo ""
echo "You can now run training with:"
echo "python train.py --data_root ${DATA_ROOT}"

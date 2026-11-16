#!/usr/bin/env bash
# Script to download and structure the HAM10000 dataset for Q-MedCAM

set -e

DATA_DIR="data"
IMAGES_DIR="data/images"

mkdir -p "$DATA_DIR"
mkdir -p "$IMAGES_DIR"

echo "====================================================================="
echo "Q-MedCAM: HAM10000 Dataset Setup"
echo "====================================================================="
echo "Target directory: $DATA_DIR"
echo ""

if command -v kaggle &> /dev/null; then
    echo "Kaggle CLI detected. Downloading dataset..."
    kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$DATA_DIR" --unzip
    
    if [ -d "$DATA_DIR/HAM10000_images_part_1" ]; then
        echo "Consolidating part 1 images..."
        cp "$DATA_DIR"/HAM10000_images_part_1/*.jpg "$IMAGES_DIR"/ 2>/dev/null || true
    fi
    if [ -d "$DATA_DIR/HAM10000_images_part_2" ]; then
        echo "Consolidating part 2 images..."
        cp "$DATA_DIR"/HAM10000_images_part_2/*.jpg "$IMAGES_DIR"/ 2>/dev/null || true
    fi
    echo "Dataset setup complete!"
elif python3 -c "import kagglehub" &> /dev/null; then
    echo "Kagglehub detected. Downloading public HAM10000 dataset..."
    KAGGLE_PATH=$(python3 -c "import kagglehub; print(kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000'))")
    echo "Downloaded to: $KAGGLE_PATH"
    cp "$KAGGLE_PATH/HAM10000_metadata.csv" "$DATA_DIR/"
    ln -sf "$KAGGLE_PATH"/HAM10000_images_part_1/*.jpg "$IMAGES_DIR/" 2>/dev/null || true
    ln -sf "$KAGGLE_PATH"/HAM10000_images_part_2/*.jpg "$IMAGES_DIR/" 2>/dev/null || true
    echo "Dataset setup complete via kagglehub!"
else
    echo "Neither Kaggle CLI nor kagglehub found."
    echo "To download manually:"
    echo "1. Download HAM10000 from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
    echo "   or https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
    echo "2. Place HAM10000_metadata.csv into: data/HAM10000_metadata.csv"
    echo "3. Extract all lesion images (.jpg) into: data/images/"
fi

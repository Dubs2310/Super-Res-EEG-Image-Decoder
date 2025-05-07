#!/usr/bin/env bash

URL="https://files.osf.io/v1/resources/kqgs8\
/providers/osfstorage/6628009bd896070b9f1b15c8\
/?view_only=3274f4ce973e4240b01df955445daad6&zip="

DIR="$(dirname "$0")"
OUT_DIR="$DIR/eeg/raw"
ZIP_DIR="$DIR/eeg/raw.zip"
PYTHON_DIR="$DIR/../../code/scripts"

mkdir -p $OUT_DIR

if [ ! "$(ls -A "$OUT_DIR/")" ]; then

    curl -L -o "$ZIP_DIR" "$URL"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully."
        unzip -q "$ZIP_DIR" -d "$OUT_DIR"

        if [ $? -eq 0 ]; then
            echo "Raw EEG Data extracted successfully to $OUT_DIR"
        else
            echo "Failed to extract the ZIP file."
            exit 1
        fi

        rm "$ZIP_DIR"
    else
        echo "Download failed."
        exit 1
    fi
fi

mkdir "$DIR/eeg/preprocessed"

echo "Preprocessing Data.."
python "$PYTHON_DIR/preprocess_raw.py"
if [ $? -eq 0 ]; then
    echo "Successfully Preprocessed"

    echo "Building Comprehensive Dataset.."
    python "$PYTHON_DIR/build_comprehensive_dataset.py"
    if [ $? -eq 0 ]; then
        echo "Successfully built comprehensive dataset"
    fi
fi
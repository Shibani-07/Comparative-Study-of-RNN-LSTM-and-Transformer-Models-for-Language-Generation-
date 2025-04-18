#!/bin/bash

# train_all_models.sh

# Set base directory
BASE_DIR="./data"

# Create necessary directories if they don't exist
mkdir -p "$BASE_DIR/plots"
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/logs"

# Function to print section headers
print_header() {
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# Common training parameters
COMMON_PARAMS="--base_dir $BASE_DIR \
    --dim_embed 384 \
    --dim_hidden 512 \
    --num_layers 4 \
    --num_heads 4 \
    --p_drop 0.1 \
    --batch_size 128 \
    --epochs 30 \
    --lr 5e-3 \
    --max_len 512 \
    --patience 5"

# Function to train a model
train_model() {
    MODEL_TYPE=$1
    print_header "Training $MODEL_TYPE Model"
    echo "Starting training at $(date)"
    
    # Run training and save logs
    python train.py --model_type $MODEL_TYPE $COMMON_PARAMS 2>&1 | tee "$BASE_DIR/logs/${MODEL_TYPE}_training.log"
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "$MODEL_TYPE model training completed successfully"
        # Move model file to models directory
        mv "$BASE_DIR/${MODEL_TYPE}_model.pt" "$BASE_DIR/models/" 2>/dev/null
    else
        echo "Error: $MODEL_TYPE model training failed"
    fi
    
    echo "Finished at $(date)"
    echo
}

# Main execution
print_header "Starting Training Pipeline"
echo "Training will be performed for transformer, lstm, and rnn models"
echo "Logs will be saved in $BASE_DIR/logs"
echo "Models will be saved in $BASE_DIR/models"
echo "Plots will be saved in $BASE_DIR/plots"
echo

# Check if Python and required packages are available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check if required Python packages are installed
python -c "import torch; import tqdm; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages are not installed"
    echo "Please install required packages using:"
    echo "pip install torch tqdm matplotlib seaborn"
    exit 1
fi

# Train all models
for MODEL_TYPE in transformer lstm rnn; do
    train_model $MODEL_TYPE
done

print_header "Training Pipeline Completed"
echo "Summary:"
echo "- Models saved in: $BASE_DIR/models/"
echo "- Training logs saved in: $BASE_DIR/logs/"
echo "- Training plots saved in: $BASE_DIR/plots/"

# Print training times from logs
echo -e "\nTraining Times:"
for MODEL_TYPE in transformer lstm rnn; do
    if [ -f "$BASE_DIR/logs/${MODEL_TYPE}_training.log" ]; then
        START_TIME=$(head -n 1 "$BASE_DIR/logs/${MODEL_TYPE}_training.log" | grep -o "Starting training at.*")
        END_TIME=$(tail -n 2 "$BASE_DIR/logs/${MODEL_TYPE}_training.log" | grep -o "Finished at.*")
        echo "$MODEL_TYPE:"
        echo "  $START_TIME"
        echo "  $END_TIME"
    fi
done

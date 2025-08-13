#!/bin/bash

# --------------------
# Read .env 
# --------------------
set -a
source .env
set +a



# --------------------
# Configuration
# --------------------
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"                   # HuggingFace model name or local path to the model
NUM_GPUS=2                                                      # Number of GPUs to use
GPU_DEVICES="device=0,1"                                        # List of GPU device IDs to assign to Docker
GPU_MEMORY_UTILIZATION=0.95                                      # Maximum GPU memory utilization ratio (between 0 and 1)
MAX_LEN=16000                                                   # Maximum token length
HOST="0.0.0.0"                                                  # Host IP to bind the vLLM server (0.0.0.0 binds all interfaces)
HOST_PORT=8000                                                  # Port on the host machine to expose
CONTAINER_NAME="vllm_server"                                    # Docker container name (should be unique)
LOCAL_HF_MODEL_DIR="/home/data_storage/huggingface"             # Diretory which contains HuggingFace models



# --------------------
# Docker command
# --------------------
CMD="--model $MODEL_NAME \
     --tensor-parallel-size $NUM_GPUS \
     --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
     --max-model-len $MAX_LEN \
     --host $HOST \
     --port $HOST_PORT"



# --------------------
# Docker execution
# --------------------
echo "[INFO] Starting vLLM server using Docker..."

docker run --gpus "\"$GPU_DEVICES\"" --rm -d \
           --name $CONTAINER_NAME \
           -p $HOST_PORT:8000 \
           --ipc=host \
           -v $LOCAL_HF_MODEL_DIR:/root/.cache/huggingface/hub/ \
           --env HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
           vllm/vllm-openai:latest \
           $CMD

echo "[INFO] vLLM Docker container '$CONTAINER_NAME' is running on port $HOST_PORT"
echo "[INFO] You can see the logs using the following command: docker logs --tail 1000 -f $CONTAINER_NAME"

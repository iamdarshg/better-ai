#!/bin/bash

# Distributed training launch script for Better AI
# Supports single-node multi-GPU and multi-node training

GPUS=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
NNODES=1
USE_DEEPSPEED=false
DS_CONFIG="configs/deepspeed_zero3.json"
TRAIN_CONFIG="configs/training_do_single_h100.yml"
STAGE="pretrain"
EXTRA_ARGS=""

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --gpus N             Number of GPUs per node (default: 1)"
    echo "  --nnodes N           Number of nodes (default: 1)"
    echo "  --node_rank N        Rank of this node (default: 0)"
    echo "  --master_addr ADDR   Address of master node (default: localhost)"
    echo "  --master_port PORT   Port of master node (default: 29500)"
    echo "  --deepspeed          Enable DeepSpeed (default: false)"
    echo "  --ds_config PATH     Path to DeepSpeed config (default: configs/deepspeed_zero3.json)"
    echo "  --config PATH        Path to training config YAML/JSON (default: configs/training_do_single_h100.yml)"
    echo "  --stage STAGE        Training stage (pretrain, sft, rlhf, security_dpo) (default: pretrain)"
    echo "  --help               Show this help message"
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --deepspeed)
            USE_DEEPSPEED=true
            shift
            ;;
        --ds_config)
            DS_CONFIG="$2"
            shift 2
            ;;
        --config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$((NNODES * GPUS))

echo "Launching distributed training:"
echo "  Nodes: $NNODES"
echo "  GPUs per node: $GPUS"
echo "  World Size: $WORLD_SIZE"
echo "  Node Rank: $NODE_RANK"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Stage: $STAGE"
echo "  Train Config: $TRAIN_CONFIG"
if [ "$USE_DEEPSPEED" = true ]; then
    echo "  DeepSpeed: Enabled (Config: $DS_CONFIG)"
fi

CMD="torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    better_ai/scripts/main_workflow.py \
    --stage $STAGE \
    --config $TRAIN_CONFIG \
    $EXTRA_ARGS"

if [ "$USE_DEEPSPEED" = true ]; then
    CMD="$CMD --use-deepspeed --deepspeed-config $DS_CONFIG"
fi

echo "Running command: $CMD"
$CMD

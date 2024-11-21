#!/bin/bash
set -e

HOST="0.0.0.0"
PORT=8001
MODEL_PATH="/root/paddlejob/workspace/env_run/luyao15/weights/Doc-Lark"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python paddlemix/examples/qwen2_vl/server.py --host "$HOST" --port "$PORT" --model-path "$MODEL_PATH"
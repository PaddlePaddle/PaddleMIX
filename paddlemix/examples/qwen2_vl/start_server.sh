#!/bin/bash
set -e

HOST="localhost"
PORT=8080
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"

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
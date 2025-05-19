docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --served-model-name qwen-coder-32b \
    --quantization fp8
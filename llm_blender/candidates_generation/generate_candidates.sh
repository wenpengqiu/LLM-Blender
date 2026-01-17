# <===================== Generation for mixed using multiple models =====================>
# 1. 定义选择的 6 个 SLM
models=(
    "deepseek-ai/deepseek-llm-7b-chat"
    "tiiuae/Falcon3-10B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "Qwen/Qwen2-7B-Instruct"
    "StabilityAI/StableLM-Zephyr-3B"
)

# 2. 设置要为其生成数据的数据集和子集
# 需要为 mmlu(test, val) 和 gsm8k(train, test) 分别运行
dataset="mmlu"
set="test"

prompt_max_length=1536
output_max_length=1536

cmd="bash"

# 3. 循环启动任务
for model in "${models[@]}"; do
    echo "Starting generation for model: $model"
    echo "Dataset: $dataset, Set: $set"
    ${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"
done

echo "All generation tasks for $dataset $set submitted."
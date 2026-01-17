import os
import json
import torch
import numpy as np
from tqdm import tqdm
from llm_blender.pair_ranker.model_util import build_ranker, build_tokenizer
from llm_blender.pair_ranker.config import RankerConfig

# 1. 当前任务: 修改这里切换 'gsm8k' 或 'mmlu'
CURRENT_TASK = "gsm8k"  # 选项: "gsm8k" 或 "mmlu"

# 2. 基础模型配置
MODEL_TYPE = "deberta" 
MODEL_NAME = "microsoft/deberta-v3-large" 
TOP_K = 3            # 选最好的3个给GenFuser
EVAL_BATCH_SIZE = 16  # 推理时的Batch Size

# 3. 路径自动配置 (根据 CURRENT_TASK 自动选择)
if CURRENT_TASK == "gsm8k":
    # GSM8K 配置
    RANKER_CHECKPOINT = "/data2/qwp/LLM-Blender/checkpoints/ranker_gsm8k/checkpoint-best"       # Ranker 权重路径
    DATA_DIR = "/data2/qwp/LLM-Blender/data/pair_ranker_data/gsm8k"             # 输入数据路径
    OUTPUT_DIR = "/data2/qwp/LLM-Blender/data/gen_fuser_data/gsm8k"             # 输出保存路径
    MAX_LEN = 1280
    
elif CURRENT_TASK == "mmlu":
    # MMLU 配置
    RANKER_CHECKPOINT = "/data2/qwp/LLM-Blender/checkpoints/ranker_mmlu/checkpoint-best"
    DATA_DIR = "/data2/qwp/LLM-Blender/data/pair_ranker_data/mmlu"
    OUTPUT_DIR = "/data2/qwp/LLM-Blender/data/gen_fuser_data/mmlu"
    MAX_LEN = 1280

else:
    raise ValueError(f"Unknown task: {CURRENT_TASK}")

print(f"==================================================")
print(f">>> 当前任务: {CURRENT_TASK}")
print(f">>> Ranker路径: {RANKER_CHECKPOINT}")
print(f">>> 输入数据: {DATA_DIR}")
print(f">>> 输出路径: {OUTPUT_DIR}")
print(f">>> Max Length: {MAX_LEN}")
print(f"==================================================")

# ==============================================================================

def load_data_from_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return []
        
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'candidates' in item and len(item['candidates']) >= 2:
                data.append(item)
    return data

def get_pairwise_scores(model, tokenizer, item, device):
    """
    使用模型对候选答案进行全量两两比较 (Full Comparison)，计算全局得分
    """
    source = item['source']
    candidates = [c['text'] for c in item['candidates']]
    n = len(candidates)
    
    scores_matrix = np.zeros((n, n))
    
    # 准备 Batch
    batch_source = []
    batch_cand1 = []
    batch_cand2 = []
    pair_indices = []

    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # 格式对齐 Ranker 训练时的处理
            batch_source.append(source)
            batch_cand1.append(candidates[i])
            batch_cand2.append(candidates[j])
            pair_indices.append((i, j))

    # 分批处理以防 OOM
    
    model.eval()
    with torch.no_grad():
        for k in range(0, len(batch_source), EVAL_BATCH_SIZE):
            b_src = batch_source[k : k + EVAL_BATCH_SIZE]
            b_c1 = batch_cand1[k : k + EVAL_BATCH_SIZE]
            b_c2 = batch_cand2[k : k + EVAL_BATCH_SIZE]
            
            # Tokenize
            enc_source = tokenizer(b_src, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
            enc_c1 = tokenizer(b_c1, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
            enc_c2 = tokenizer(b_c2, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
            
            # 兼容性处理：检查模型是否有 _forward 方法 (LLM-Blender Wrapper)
            if hasattr(model, "_forward"):
                outputs = model._forward(
                    source_ids=enc_source['input_ids'],
                    source_attention_mask=enc_source['attention_mask'],
                    cand1_ids=enc_c1['input_ids'],
                    cand1_attention_mask=enc_c1['attention_mask'],
                    cand2_ids=enc_c2['input_ids'],
                    cand2_attention_mask=enc_c2['attention_mask']
                )
                # outputs 包含 'logits' (i 优于 j 的分数)
                batch_scores = outputs['logits'].cpu().numpy() # [batch_size]
            else:
                # 原生模型兜底 (一般不会走到这，除非你没用 build_ranker)
                raise ValueError("Model is not a CrossCompareReranker instance!")
            
            # 填入矩阵
            for idx, score in enumerate(batch_scores):
                i, j = pair_indices[k + idx]
                scores_matrix[i, j] = score

    # 计算全局得分 (Net Win Rate)
    # Score_i = sum(Score_ij) - sum(Score_ji)
    candidate_scores = np.sum(scores_matrix, axis=1) - np.sum(scores_matrix, axis=0)
    
    ranked_indices = np.argsort(candidate_scores)[::-1]
    return ranked_indices[:TOP_K]

def process_and_save(split_name, model, tokenizer, device):
    print(f"Processing {split_name} set from {DATA_DIR}...")
    input_file = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    
    raw_data = load_data_from_jsonl(input_file)
    if not raw_data:
        return

    fuser_data = []
    
    # 使用 tqdm 显示进度
    for item in tqdm(raw_data, desc=f"Generating {split_name}"):
        try:
            top_indices = get_pairwise_scores(model, tokenizer, item, device)
            
            all_candidates = item['candidates']
            top_k_texts = [all_candidates[i]['text'] for i in top_indices]
            
            source_text = item['source']
            target_text = item.get('target', '') 
            
            # 构造 GenFuser 的 Input Prompt
            # 格式: Question ... \n Candidates: \n [1] ... \n [2] ...
            fuser_input = source_text + "\n\nHere are some candidate solutions:\n"
            for k, cand_text in enumerate(top_k_texts):
                fuser_input += f"\n[Candidate {k+1}]:\n{cand_text}\n"
            fuser_input += "\nBased on the above candidates, provide the final correct answer:"
            
            fuser_data.append({
                "input": fuser_input,
                "output": target_text
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM detected, skipping sample...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in fuser_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(fuser_data)} samples to {output_file}")

def fix_tokenizer(tokenizer):
    """
    修复 Tokenizer 缺失的属性。
    LLM-Blender 训练时动态添加了这些属性，但加载时会丢失，需要手动补回。
    """
    # 定义特殊 Token
    special_tokens_dict = {
        "source_prefix": "<|source|>",
        "candidate1_prefix": "<|candidate1|>",
        "candidate2_prefix": "<|candidate2|>",
        "candidate_prefix": "<|candidate|>"
    }
    
    # 确保这些 Token 在词表中 (训练过的 checkpoint 应该已经有了，这一步是双重保险)
    tokenizer.add_tokens(list(special_tokens_dict.values()))
    
    # 关键步骤：手动设置属性
    tokenizer.source_prefix_id = tokenizer.convert_tokens_to_ids("<|source|>")
    tokenizer.cand1_prefix_id = tokenizer.convert_tokens_to_ids("<|candidate1|>")
    tokenizer.cand2_prefix_id = tokenizer.convert_tokens_to_ids("<|candidate2|>")
    tokenizer.cand_prefix_id = tokenizer.convert_tokens_to_ids("<|candidate|>")
    
    return tokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Ranker from {RANKER_CHECKPOINT}...")
    
    # 1. 加载 Config
    config_path = os.path.join(RANKER_CHECKPOINT, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. Did the training finish?")
        
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = RankerConfig()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
            
    # 2. 加载 Tokenizer
    tokenizer = build_tokenizer(RANKER_CHECKPOINT, cache_dir=None)
    
    tokenizer = fix_tokenizer(tokenizer)
    print("Tokenizer attributes fixed (source_prefix_id, cand_prefix_id, etc.)")
    
    # 3. 加载 Model Structure
    model = build_ranker(
        ranker_type="pairranker",
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        cache_dir=None,
        config=config,
        tokenizer=tokenizer
    )
    
    # 4. 加载 Weights
    bin_path = os.path.join(RANKER_CHECKPOINT, "pytorch_model.bin")
    safetensors_path = os.path.join(RANKER_CHECKPOINT, "model.safetensors")
    
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        print(f"Loading weights from {safetensors_path}...")
        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(bin_path):
        print(f"Loading weights from {bin_path}...")
        state_dict = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(f"No model weights found in {RANKER_CHECKPOINT}")
        
    # 处理可能的 Key 不匹配 (比如 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 5. 处理数据
    process_and_save("train", model, tokenizer, device)
    process_and_save("val", model, tokenizer, device)
    
    print(f"Done! GenFuser data for task [{CURRENT_TASK}] is ready in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

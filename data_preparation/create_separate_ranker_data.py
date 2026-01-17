import os
import json
import re
import random

# ==========================================
# 1. 提取函数 (保持不变)
# ==========================================

def extract_mmlu_answer(text):
    if not text: return None
    parts = re.split(r'Final Answer[:\s]*', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        target_text = parts[-1].strip()
        match = re.search(r'^[\(\[]?([A-D])[\)\]\.]?', target_text)
        if match: return match.group(1)
    matches = re.findall(r'[\(\s]([A-D])[\)\.]', text)
    if matches: return matches[-1]
    matches = re.findall(r'\b([A-D])\b', text)
    if matches: return matches[-1]
    return None

def extract_gsm8k_answer(text):
    if not text: return None
    def parse_num(num_str):
        try:
            clean_str = num_str.replace(',', '')
            if clean_str.endswith('.'): clean_str = clean_str[:-1]
            if '.' in clean_str: return float(clean_str)
            return int(clean_str)
        except: return None
    if "####" in text:
        parts = text.split("####")
        match = re.search(r'(-?[\d,]+(?:\.\d+)?)', parts[-1])
        if match: return parse_num(match.group(1))
    parts = re.split(r'Final Answer[:\s]', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        target_text = parts[-1]
        matches = re.findall(r'-?[\d,]+(?:\.\d+)?', target_text)
        if matches: return parse_num(matches[-1])
    matches = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if matches: return parse_num(matches[-1])
    return None

# ==========================================
# 2. 数据处理逻辑 (核心处理函数)
# ==========================================

def process_dataset(dataset_name, split, data_root, models):
    print(f"Processing {dataset_name} - {split}...")
    
    # 1. 读取 Ground Truth
    # 假设路径结构: ../data/gsm8k/train_data.json
    data_file = os.path.join(data_root, dataset_name, f"{split}_data.json")
    if not os.path.exists(data_file):
        print(f"Error: GT file not found: {data_file}")
        return []

    with open(data_file, 'r') as f:
        gt_data = json.load(f)
    id_to_gt = {item['id']: item for item in gt_data}
    
    # 2. 读取所有模型候选，整理为 {id: [candidate_obj, ...]}
    aggregated_data = {} # {id: {"source": ..., "candidates": []}}
    
    for model in models:
        safe_model_name = model.split('/')[-1]
        # 假设路径结构: ../data/gsm8k/candidates/train/beam_search/model_name.jsonl
        candidate_file = os.path.join(data_root, dataset_name, "candidates", split, "beam_search", f"{safe_model_name}.jsonl")
        
        if not os.path.exists(candidate_file):
            print(f"Warning: Candidate file not found: {candidate_file}")
            continue
            
        print(f"  Loading {safe_model_name} from {split}...")
        with open(candidate_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q_id = item['id']
                    if q_id not in id_to_gt: continue
                    
                    # 初始化该问题的数据结构
                    if q_id not in aggregated_data:
                        gt_item = id_to_gt[q_id]
                        prompt = gt_item['instruction'] + "\n" + gt_item['input']
                        aggregated_data[q_id] = {
                            "source": prompt,
                            "target": str(gt_item['ground_truth']), # 保存标准答案以备查
                            "candidates": []
                        }
                    
                    # 获取文本并评分
                    text = item['candidates'][0]['text']
                    ground_truth = id_to_gt[q_id]['ground_truth']
                    
                    is_correct = False
                    if dataset_name == 'mmlu':
                        pred = extract_mmlu_answer(text)
                        if pred and str(pred).upper() == str(ground_truth).upper():
                            is_correct = True
                    else: # gsm8k
                        pred = extract_gsm8k_answer(text)
                        try:
                            gt_str = str(ground_truth)
                            clean_gt = float(gt_str.split("####")[-1].strip().replace(',', '')) if "####" in gt_str else float(gt_str.strip().replace(',', ''))
                            if pred is not None and abs(pred - clean_gt) < 1e-6:
                                is_correct = True
                        except: pass
                    
                    # 构造符合 LLM-Blender 原生格式的 candidate
                    candidate_entry = {
                        "text": text,
                        "scores": {
                            "accuracy": 1.0 if is_correct else 0.0
                        },
                        "model": safe_model_name
                    }
                    aggregated_data[q_id]["candidates"].append(candidate_entry)
                    
                except json.JSONDecodeError: continue

    # 转换为列表并过滤掉候选数不足的样本
    final_list = []
    for q_id, data in aggregated_data.items():
        if len(data["candidates"]) >= 2: # 至少要有2个候选才能比较
            final_list.append(data)
            
    print(f"  > Loaded {len(final_list)} valid samples for {dataset_name}-{split}")
    return final_list

# ==========================================
# 3. 主执行逻辑 (修改了GSM8K的划分方式)
# ==========================================

if __name__ == "__main__":
    data_root = "../data" 
    models = [
        "deepseek-ai/deepseek-llm-7b-chat",
        "tiiuae/Falcon3-10B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Qwen/Qwen2-7B-Instruct",
        "StabilityAI/StableLM-Zephyr-3B"
    ]

    # --- 1. 处理 GSM8K (不再随机切分，Train做训练，Test做验证) ---
    print("\n>>> Processing GSM8K (Full Train / Full Test Mode)...")
    
    # 1.1 处理 Train 集 (读取 gsm8k/train)
    gsm8k_train_data = process_dataset("gsm8k", "train", data_root, models)
    
    # 1.2 处理 Val 集 (读取 gsm8k/test) -> 保存为 val.jsonl
    gsm8k_val_data = process_dataset("gsm8k", "test", data_root, models)
    
    # 1.3 保存
    output_gsm8k_dir = os.path.join(data_root, "pair_ranker_data/gsm8k")
    os.makedirs(output_gsm8k_dir, exist_ok=True)
    
    with open(os.path.join(output_gsm8k_dir, "train.jsonl"), 'w') as f:
        for item in gsm8k_train_data: f.write(json.dumps(item) + "\n")
    print(f"Saved {len(gsm8k_train_data)} samples to {output_gsm8k_dir}/train.jsonl")
        
    with open(os.path.join(output_gsm8k_dir, "val.jsonl"), 'w') as f:
        for item in gsm8k_val_data: f.write(json.dumps(item) + "\n")
    print(f"Saved {len(gsm8k_val_data)} samples to {output_gsm8k_dir}/val.jsonl")


    # --- 2. 处理 MMLU (保持原逻辑：切分Test集) ---
    print("\n>>> Processing MMLU (Split Test Mode)...")
    # MMLU 依然读取 test，然后切分，因为没有标准的验证集文件结构
    mmlu_data = process_dataset("mmlu", "test", data_root, models)
    
    # 随机打乱并切分 80/20
    random.shuffle(mmlu_data)
    split_idx = int(len(mmlu_data) * 0.8)
    
    output_mmlu_dir = os.path.join(data_root, "pair_ranker_data/mmlu")
    os.makedirs(output_mmlu_dir, exist_ok=True)
    
    with open(os.path.join(output_mmlu_dir, "train.jsonl"), 'w') as f:
        for item in mmlu_data[:split_idx]: f.write(json.dumps(item) + "\n")
    print(f"Saved {split_idx} samples to {output_mmlu_dir}/train.jsonl")

    with open(os.path.join(output_mmlu_dir, "val.jsonl"), 'w') as f:
        for item in mmlu_data[split_idx:]: f.write(json.dumps(item) + "\n")
    print(f"Saved {len(mmlu_data) - split_idx} samples to {output_mmlu_dir}/val.jsonl")

    print("\nDone! Data separated into ./data/pair_ranker_data/gsm8k and ./data/pair_ranker_data/mmlu")
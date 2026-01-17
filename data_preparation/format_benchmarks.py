import os
import json
from datasets import load_dataset
from tqdm import tqdm

def format_mmlu_prompt(sample):
    """
    修改 MMLU prompt，强制要求 CoT 和标准化的输出格式。
    """
    choices = ["(A)", "(B)", "(C)", "(D)"]
    instruction = "The following is a multiple-choice question. Please provide a step-by-step reasoning for your answer, and then conclude with the final answer on a new line in the format: 'Final Answer: (X)'."
    
    question_body = sample['question']
    for i, choice in enumerate(sample['choices']):
        question_body += f"\n{choices[i]} {choice}"
    
    # 将格式化的 prompt 放入 'instruction'， 'input' 留空
    # generate_candidates.py 会自动拼接 instruction 和 input
    final_prompt = f"Instruction:\n{instruction}\n\nQuestion:\n{question_body}"
    return final_prompt

def create_gsm8k_data(data_dir):
    """
    加载并格式化 GSM8K 数据集
    """
    print("Processing GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    for split in ['train', 'test']:
        output_data = []
        for i, sample in tqdm(enumerate(dataset[split]), desc=f"Formatting GSM8K {split}"):
            # llm-blender 期望 'instruction' 和 'input'
            output_data.append({
                "id": f"gsm8k_{split}_{i}",
                "instruction": f"Solve the following math problem. Show your step-by-step reasoning. Conclude with the final answer on a new line in the format: 'Final Answer: XXX'.\n\nProblem:\n{sample['question']}",
                "input": "", # GSM8K 问题是自包含的
                "ground_truth": sample['answer'] # 保存标准答案以供后续训练
            })
        
        # 按照 llm-blender 的命名习惯保存
        split_name = "val" if split == "test" else "train" # GSM8K 没有 val, 我们用 test 代替
        output_dir = os.path.join(data_dir, "gsm8k")
        os.makedirs(output_dir, exist_ok=True)
        # llm_blender/candidates_generation/generate_candidates.py (line 307) 期望 {set_name}_data.json
        output_file = os.path.join(output_dir, f"{split_name}_data.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    print("GSM8K formatting done.")

def create_mmlu_data(data_dir):
    """
    加载并格式化 MMLU 数据集
    """
    print("Processing MMLU...")
    # MMLU 需要从 'cais/mmlu' 加载所有子任务
    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    # MMLU 分为 validation (用作 few-shot) 和 test (主要评估) 和 dev (训练)
    # llm-blender 期望 'train', 'val', 'test'
    # 我们将 MMLU 的 'dev' -> 'train', 'validation' -> 'val', 'test' -> 'test'
    split_mapping = {'dev': 'train', 'validation': 'val', 'test': 'test'}
    choice_map = ["A", "B", "C", "D"]

    for mmlu_split, blender_split in split_mapping.items():
        output_data = []
        for subject in tqdm(subjects, desc=f"Formatting MMLU {blender_split}"):
            try:
                dataset = load_dataset("cais/mmlu", subject, split=mmlu_split)
            except Exception as e:
                print(f"Could not load {subject} for split {mmlu_split}: {e}")
                continue
                
            for i, sample in enumerate(dataset):
                formatted_prompt = format_mmlu_prompt(sample)
                output_data.append({
                    "id": f"mmlu_{subject}_{blender_split}_{i}",
                    "instruction": formatted_prompt,
                    "input": "",
                    "ground_truth": choice_map[sample['answer']] # 保存 A/B/C/D 格式的答案
                })
        
        output_dir = os.path.join(data_dir, "mmlu")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{blender_split}_data.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    print("MMLU formatting done.")

if __name__ == "__main__":
    # 定义您的数据根目录 (与 _generate_candidates.sh 中的 $data_dor 保持一致)
    data_root = "../data" 
    
    # 确保根目录存在
    os.makedirs(data_root, exist_ok=True)
    
    create_gsm8k_data(data_root)
    create_mmlu_data(data_root)
    print("All datasets formatted successfully.")

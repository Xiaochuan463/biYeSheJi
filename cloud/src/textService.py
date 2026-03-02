# -------------------------------
# 文本服务 / BERT 中文微调头部
# -------------------------------

# 1️⃣ 导入基础库
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

# 2️⃣ 检查GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 3️⃣ 下载BERT中文基础模型
# 用于意图分类 (单标签分类)
local_model_path = r"D:\Doc\ComDes\cloud\src\bert-base-chinese"  # 替换为你的本地模型路径
intent_model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 意图分类模型
intent_model = AutoModelForSequenceClassification.from_pretrained(
    local_model_path,
    num_labels=5,               # 你的意图数量
    problem_type="single_label" # 单标签分类
).to(device)

# 如果做槽位填充 (序列标注任务)
slot_model_name = "bert-base-chinese"
slot_tokenizer = AutoTokenizer.from_pretrained(slot_model_name)

# 假设槽位标签数量为 N
num_slot_labels = 3  # 根据你自己的标签数
slot_model = AutoModelForTokenClassification.from_pretrained(
    slot_model_name,
    num_labels=num_slot_labels
).to(device)

# 4️⃣ 可选：启用 gradient checkpointing 节省显存
intent_model.gradient_checkpointing_enable()
slot_model.gradient_checkpointing_enable()

print("模型与分词器加载完成 ✅")
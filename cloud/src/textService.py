import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intend_model_path = "D:/Doc/biYeSheJi/cloud/src/models/intent_model"
slot_model_path = "D:/Doc/biYeSheJi/cloud/src/models/slot_model"
tokenizer_path = "D:/Doc/biYeSheJi/cloud/src/models/tokenizer"

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
intend_model = BertForSequenceClassification.from_pretrained(intend_model_path)
slot_model = BertForTokenClassification.from_pretrained(slot_model_path)

intend_model.to(device)
slot_model.to(device)
intend_model.eval()
slot_model.eval()

slot_labels = ['o', 'b', 'i']
slot2id = {label: idx for idx, label in enumerate(slot_labels)}
id2slot = {idx: label for idx, label in enumerate(slot_labels)}
intent_labels = ['B', 'F', 'L', 'R', 'S', 'N']
id2intent = {0: 'B', 1: 'F', 2: 'L', 3: 'R', 4: 'S', 5: 'N'}
intent2id = {v: k for k, v in id2intent.items()}
is_single_intent = len(intent2id) == 1

def predict(text):
    # 预处理输入
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        max_length=30,
        truncation=True
    ).to(device)
    
    # 关闭梯度计算
    with torch.no_grad():
        # 1. 槽位预测
        slot_logits = slot_model(**inputs).logits
        slot_preds = torch.argmax(slot_logits, dim=-1).squeeze().cpu().numpy()
        
        # 2. 意图预测
        intent_logits = intend_model(**inputs).logits
        if is_single_intent:
            # 单意图直接返回唯一标签
            intent_label = intent_labels[0]
        else:
            intent_pred = torch.argmax(intent_logits, dim=-1).item()
            intent_label = id2intent[intent_pred]
    
    # 解析槽位（过滤特殊token）
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    slot_results = []
    for token, pred in zip(tokens, slot_preds):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        slot_label = id2slot[pred] if pred != -100 else 'o'
        slot_results.append((token, slot_label))
    
    return {
        '输入文本': text,
        '预测意图': intent_label,
        '槽位预测': slot_results
    }

print("模型加载完成")

test_text = "呱，是敌人，快撤"
result = predict(test_text)
print("===== 推理结果 =====")
print(f"输入文本：{result['输入文本']}")
print(f"预测意图：{result['预测意图']}")
print(f"槽位预测：{result['槽位预测']}")
import json

data = []
with open('data/train.json', 'r', encoding='utf-8') as f, open('processed_data/train.json', 'w', encoding='utf-8') as fw:
    for line in f:
        line = json.loads(line)
        data.append(line)
    json.dump(data, fw, ensure_ascii=False, indent=4)

data = []
with open('data/validation-rt.json', 'r', encoding='utf-8') as f, open('processed_data/dev.json', 'w', encoding='utf-8') as fw:
    for line in f:
        line = json.loads(line)
        data.append(line)
    json.dump(data, fw, ensure_ascii=False, indent=4)

data = []
with open('data/test.jsonl', 'r', encoding='utf-8') as f, open('processed_data/test.json', 'w', encoding='utf-8') as fw:
    for line in f:
        line = json.loads(line)
        data.append(line)
    json.dump(data, fw, ensure_ascii=False, indent=4)

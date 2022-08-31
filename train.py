import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import random
from tqdm import tqdm
import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class Config:
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 16
    TRAIN_EPOCHS = 100
    LEARNING_RATE = 1e-4
    SEED = 42
    MAX_IN_LEN = 1500  # TODO
    MAX_OUT_LEN = 200  # TODO
    ACCUM_ITER = 1
    WEIGHT_DECAY = 1e-2
    MODEL = "google/mt5-base"  # TODO
    BEAMS = 3
    do_train = True
    do_test = False
    patience = 3
    repetition_penalty = 23.0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CFG = Config()
seed_everything(CFG.SEED)  # 固定随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        each = self.data[index]
        content = each['content'].replace(' ', '')
        if 'summary' in each:  # train
            summary = each['summary'].replace(' ', '')
        else:  # test
            summary = ''
        search_qa = each['retrieval']['search_qa']
        source = content
        for qa in search_qa:
            source += '<|question|>' + qa['question'] + '<|answer|>' + qa['answer']

        source = self.tokenizer(source, max_length=CFG.MAX_IN_LEN, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = source['input_ids'].squeeze(0)
        attention_mask = source["attention_mask"].squeeze(0)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(summary, max_length=CFG.MAX_OUT_LEN, padding='max_length', truncation=True, return_tensors='pt')
        labels = labels["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, tokenizer, model, device, loader, optimizer):
    print('epoch:', epoch)
    model.train()
    losses = AverageMeter()
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    for _, data in enumerate(tk):
        inputs = {k: v.to(device) for k, v in data.items()}
        outputs = model(**inputs)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item() * CFG.ACCUM_ITER, inputs['input_ids'].size(0))
        tk.set_postfix(loss=losses.avg)


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    losses = AverageMeter()
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    with torch.no_grad():
        for _, data in enumerate(tk):
            inputs = {k: v.to(device) for k, v in data.items()}
            outputs = model(**inputs)
            loss = outputs[0]

            losses.update(loss.item() * CFG.ACCUM_ITER, inputs['input_ids'].size(0))
            tk.set_postfix(loss=losses.avg)
    return losses.avg


def inference(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    with torch.no_grad():
        for _, data in enumerate(tk):
            generated_ids = model.generate(
                input_ids=data['input_ids'].to(device, dtype=torch.long),
                attention_mask=data['attention_mask'].to(device, dtype=torch.long),
                max_length=CFG.MAX_OUT_LEN,
                num_beams=CFG.BEAMS,
                repetition_penalty=CFG.repetition_penalty,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)
    return predictions


def main():
    model = MT5ForConditionalGeneration.from_pretrained(CFG.MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)

    tokenizer.add_tokens(['<|question|>', '<|answer|>'])
    model.resize_token_embeddings(len(tokenizer))

    train_params = {
        'batch_size': CFG.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 8
    }

    val_params = {
        'batch_size': CFG.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 8
    }
    
    test_params = {
        'batch_size': CFG.TEST_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 8
    }

    if CFG.do_train:
        with open('processed_data/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open('processed_data/dev.json', 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
    
        train_set = CustomDataset(train_data, tokenizer)
        val_set = CustomDataset(dev_data, tokenizer)
        
        train_loader = DataLoader(train_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)
    
    if CFG.do_test:  
        with open('processed_data/test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        test_set = CustomDataset(test_data, tokenizer)
        test_loader = DataLoader(test_set, **test_params)

    if CFG.do_train:
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)

        best_loss = float('inf')
        for epoch in range(CFG.TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, train_loader, optimizer)
            valid_loss = validate(epoch, tokenizer, model, device, val_loader)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'model/{}.pt'.format(CFG.MODEL.split('/')[-1]))
                print('best_loss:', best_loss)
                no_update = 0
            elif valid_loss >= best_loss and (no_update < CFG.patience):
                no_update += 1
                print(f"no update: {no_update}")
            elif no_update == CFG.patience:
                print("Model has exceed patience. Exiting")
                break

    if CFG.do_test:
        model.load_state_dict(torch.load('model/{}.pt'.format(CFG.MODEL.split('/')[-1])))
        predictions = inference(tokenizer, model, device, test_loader)
        for i, each in enumerate(predictions):
            predictions[i] = each.replace(',', '，')

        import pandas as pd
        df = pd.read_csv('example.csv', index_col=0)
        for i, each in enumerate(predictions):
            df.loc[i, 'candidates'] = each
        df.to_csv('final.csv', index=True)


if __name__ == '__main__':
    main()

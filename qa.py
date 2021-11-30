import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
import pandas
from transformers import AutoTokenizer
from transformers import RobertaModel, RobertaConfig
from transformers import AdamW
import time
import argparse

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--tune', type=str, default='full', choices=['full', 'linear'])
parser.add_argument('--size', type=str, default='large', choices=['base', 'large'])
args = parser.parse_args()

save_dir = f'./result/{args.size}_{args.lr}_b{args.batch_size}'
print(save_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_type = "Roberta"
size = args.size
model_name = f"klue/roberta-{size}"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_data(path, tokenizer):
    dataset = pandas.read_csv(path,
                              delimiter='\t',
                              names=['ID', 'text', 'question', 'answer'],
                              header=0)

    tokenized = tokenizer(dataset['text'].tolist(),
                          dataset['question'].tolist(),
                          padding=True,
                          truncation=True,
                          return_tensors="pt")
    dataset['label'] = torch.tensor(dataset['answer'])
    return dataset, tokenized


class TensorDataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        label = self.labels[idx]
        return item, label

    def __len__(self):
        return len(self.labels)


base_path = './data'
train_dataset, train_tokenized = load_data(os.path.join(base_path, 'SKT_BoolQ_Train.tsv'),
                                           tokenizer)
val_dataset, val_tokenized = load_data(os.path.join(base_path, 'SKT_BoolQ_Dev.tsv'), tokenizer)

train_dataset = TensorDataset(train_tokenized, train_dataset['label'])
val_dataset = TensorDataset(val_tokenized, val_dataset['label'])

if size == 'base':
    batch_size = 16
else:
    batch_size = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class Roberta(RobertaModel):
    # Add classification layer to Roberta model
    def __init__(self, config, model_name):
        super(Roberta, self).__init__(config)
        self.roberta = RobertaModel.from_pretrained(model_name, config=config)
        self.hdim = config.hidden_size
        self.nclass = config.nclass
        # self.dropout = nn.Dropout(p=0.1)
        # self.fc1 = nn.Linear(self.hdim, self.hdim)
        # self.act = nn.Tanh()
        self.classifier = nn.Linear(self.hdim, self.nclass)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        h = outputs[0][:, 0, :]
        # h = self.act(self.fc1(self.dropout(h)))
        logits = self.classifier(h)
        return logits


config = RobertaConfig.from_pretrained(model_name)
config.nclass = 2
model = Roberta(config, model_name).to(device)

if args.tune == 'linear':
    for name, param in model.roberta.named_parameters():
        param.requires_grad = False
    print("Parameters of encoder are fixed!")

criterion = nn.CrossEntropyLoss()
lr = args.lr
num_epochs = 10

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = transformers.get_scheduler("linear",
                                       optimizer=optimizer,
                                       num_warmup_steps=num_epochs * len(train_loader) // 10,
                                       num_training_steps=num_epochs * len(train_loader))


def train_epoch(epoch, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    cor = 0
    n_sample = 0
    n_cur = 0
    s = time.time()
    optimizer.zero_grad()
    for data, target in train_loader:
        item = {key: val.to(device) for key, val in data.items()}
        target = target.to(device)

        logits = model(**item)
        loss = criterion(logits, target)

        n_cur += len(target)
        if n_cur >= args.batch_size:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n_cur = 0
        else:
            loss.backward()

        scheduler.step()
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        cor += (preds == target).sum().item()
        n_sample += len(target)

        # print(f"{cor}/{n_sample}", end='\r')

    loss_avg = total_loss / n_sample
    acc = cor / n_sample
    print(f"[Epoch {epoch}] Train loss: {loss_avg:.3f}, acc: {acc*100:.2f}, "
          f"lr: {optimizer.param_groups[0]['lr']:.6f} time: {time.time()-s:.1f}s")
    return acc


def validate(epoch, model, val_loader):
    model.eval()
    total_loss = 0
    cor = 0
    n_sample = 0
    with torch.no_grad():
        for data, target in val_loader:
            item = {key: val.to(device) for key, val in data.items()}
            target = target.to(device)

            logits = model(**item)
            loss = criterion(logits, target)
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            cor += (preds == target).sum().item()
            n_sample += len(target)

    loss_avg = total_loss / n_sample
    acc = cor / n_sample

    print(f"[Epoch {epoch}] Valid loss: {loss_avg:.3f}, acc: {acc*100:.2f}")
    return acc


best_acc = 0
for epoch in range(num_epochs):
    train_acc = train_epoch(epoch, model, train_loader, optimizer, scheduler)
    val_acc = validate(epoch, model, val_loader)
    if val_acc > best_acc:
        best_acc = val_acc

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(save_dir)

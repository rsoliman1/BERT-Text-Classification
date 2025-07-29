pip install transformers

import torch
from typing import List, Dict
import random
import numpy as np
from collections import Counter
import os
import transformers

# Set up overall seed
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class SentimentExample:
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label
        self.words = None
        self.word_indices = None

    def __repr__(self):
        return self.sentence + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()

def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    f = open(infile)
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            line = line.strip()
            fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[0] else 1
                sent = " ".join(fields[1:])
            else:
                label = 0 if "0" in fields[0] else 1
                sent = fields[1]
            sent = sent.lower()
            exs.append(SentimentExample(sent, label))
    f.close()
    return exs

def read_blind_sst_examples(infile: str) -> List[SentimentExample]:
    f = open(infile, encoding='utf-8')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            line = line.strip()
            sent = line.lower()
            exs.append(SentimentExample(sent, label=-1))
    return exs

def write_sentiment_examples(exs: List[SentimentExample], outfile: str):
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + ex.sentence + "\n")
    o.close()

train_path = "data/train.txt"
dev_path = "data/dev.txt"
blind_test_path = "data/test-blind.txt"

train_exs = read_sentiment_examples(train_path)
dev_exs = read_sentiment_examples(dev_path)
test_exs_words_only = read_blind_sst_examples(blind_test_path)
print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs_words_only)) + " train/dev/test examples")

def calculate_metrics(golds: List[int], predictions: List[int], print_only: bool=False):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    acc = float(num_correct) / num_total
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0

    print("Accuracy: %i / %i = %f" % (num_correct, num_total, acc))
    print("Precision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
          + "; Recall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
          + "; F1 (harmonic mean of precision and recall): %f" % f1)

    if not print_only:
        return acc, prec, rec, f1

from transformers import BertTokenizer, BertForSequenceClassification

pretrained_checkpoint = "textattack/bert-base-uncased-yelp-polarity"

tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

inputs = tokenizer(dev_exs[0].sentence, return_tensors="pt").to(device)

outputs = model(**inputs)

predicted_class_id = outputs.logits.argmax().item()
print("Prediction:", predicted_class_id)

class SentimentExampleBatchIterator:
    def __init__(self, data: List[SentimentExample], batch_size: int, shuffle: bool=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = None
        self._cur_idx = None

    def refresh(self):
        self._indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self._indices)
        self._cur_idx = 0

    def get_next_batch(self, tokenizer):
        if self._cur_idx < len(self.data):
            st_idx = self._cur_idx
            if self._cur_idx + self.batch_size > len(self.data) - 1:
                ed_idx = len(self.data)
            else:
                ed_idx = self._cur_idx + self.batch_size
            self._cur_idx = ed_idx
            batch_exs = [self.data[self._indices[_idx]] for _idx in range(st_idx, ed_idx)]
            batch_inputs = tokenizer(
                [ex.sentence for ex in batch_exs],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            batch_labels = torch.tensor(np.array([ex.label for ex in batch_exs], dtype=int)).to(device)
            return batch_inputs, batch_labels
        else:
            return None

def batch_predict(classifier, batch_inputs: torch.Tensor) -> List[int]:
    with torch.no_grad():
        logits = classifier(**batch_inputs).logits
    preds = logits.argmax(dim=1).tolist()
    return preds

def evaluate(classifier, exs: List[SentimentExample], return_metrics: bool=False):
    all_labels = []
    all_preds = []
    eval_batch_iterator = SentimentExampleBatchIterator(exs, batch_size=32, shuffle=False)
    eval_batch_iterator.refresh()
    batch_data = eval_batch_iterator.get_next_batch(tokenizer)
    while batch_data is not None:
        batch_inputs, batch_labels = batch_data
        all_labels += list(batch_labels)
        preds = batch_predict(classifier, batch_inputs)
        all_preds += preds
        batch_data = eval_batch_iterator.get_next_batch(tokenizer)
    if return_metrics:
        acc, prec, rec, f1 = calculate_metrics(all_labels, all_preds)
        return acc, prec, rec, f1
    else:
        calculate_metrics(all_labels, all_preds, print_only=True)

evaluate(model, dev_exs)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

import time

BATCH_SIZE = 32
N_EPOCHS = 5

batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=BATCH_SIZE, shuffle=True)

best_epoch = -1
best_acc = -1
start_time = time.time()
for epoch in range(N_EPOCHS):
    print("Epoch %i" % epoch)
    batch_iterator.refresh()
    model.train()
    batch_loss = 0.0
    batch_example_count = 0
    batch_data = batch_iterator.get_next_batch(tokenizer)
    while batch_data is not None:
        batch_inputs, batch_labels = batch_data
        model.zero_grad()
        outputs = model(**batch_inputs, labels=batch_labels.long())
        loss = outputs.loss
        batch_example_count += len(batch_labels)
        batch_loss += loss.item() * len(batch_labels)
        loss.backward()
        optimizer.step()
        batch_data = batch_iterator.get_next_batch(tokenizer)
    print("Avg loss: %.5f" % (batch_loss / batch_example_count))
    model.eval()
    acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
        model.save_pretrained("best_bert_model.ckpt")
    print("Time elapsed: %s" % time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-start_time)))
    print("-" * 10)

print("End of training! The best accuracy %.3f was obtained in epoch %d." % (best_acc, best_epoch))
model = BertForSequenceClassification.from_pretrained("best_bert_model.ckpt").to(device)

all_preds = []
eval_batch_iterator = SentimentExampleBatchIterator(test_exs_words_only, batch_size=32, shuffle=False)
eval_batch_iterator.refresh()
batch_data = eval_batch_iterator.get_next_batch(tokenizer)
while batch_data is not None:
    batch_inputs, _ = batch_data
    preds = batch_predict(model, batch_inputs)
    all_preds += preds
    batch_data = eval_batch_iterator.get_next_batch(tokenizer)

test_output_path = "data/data_to_submit/test-blind.bert-output.txt"
test_exs_predicted = [SentimentExample(ex.sentence, all_preds[ex_idx]) for ex_idx, ex in enumerate(test_exs_words_only)]
write_sentiment_examples(test_exs_predicted, test_output_path)
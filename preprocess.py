#%%
import os
import numpy as np
import pandas as pd
from torchtext.legacy import data 
import torch
import re
import nltk   
from nltk.tokenize import word_tokenize 
from torch.utils.data import Dataset
device = torch.device("cuda:0")
#%%
class CustomDataset(Dataset):
    def __init__(
        self, root, 
        train_path, test_path, predict_path,
        batch_size=64,
        valid_ratio=.2,
        max_vocab=999999,
        min_freq=1,
        shuffle=True,
    ):
        self.id = data.Field( # 학습에 쓰지 않을 column
            sequential=False, 
            use_vocab=False,
            unk_token=None
        )
        self.text = data.Field( 
            use_vocab=True,
            tokenize=word_tokenize,
            batch_first=True,
        )
        self.label = data.Field(
            sequential=False, # 0 or 1
            use_vocab=False,
            unk_token=None,
            is_target=True
        )
        train, valid = data.TabularDataset(
            path = root + train_path,
            format ='tsv',
            fields = [
                ('id', self.id),
                ('text', self.text),
                ('label', self.label)],
            skip_header=True
        ).split(split_ratio=(1 - valid_ratio))

        
        test = data.TabularDataset(
            path = root + test_path,
            format='tsv',
            fields=[
                ('id', self.id),
                ('text', self.text),
                ('label', self.label)],
            skip_header=True
        )

        
        predict = data.TabularDataset(
            path = root + predict_path,
            format='csv',
            fields=[
                ('id', self.id),
                ('text', self.text)],
            skip_header=True
        )

        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device=device,
            shuffle=shuffle,
            sort_key=lambda x: len(x.text), # 길이로 sort 후 batch 나눔!
            sort_within_batch=True, # 미니 배치 내에서 sort
        )


        self.test_loader = data.BucketIterator(
            test,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
        )


        self.predict_loader = data.BucketIterator(
            predict,
            batch_size=batch_size,
            device=device,
            shuffle=False
        )

        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq) 

"""
loaders = CustomDataset(root="C:\\Users\\hanyeji\\Desktop\\workspace\\sentiment analysis\\mycode",
    train_path='\\ratings_train.txt',test_path='\\ratings_train.txt',
    predict_path='\\ko_data.csv',
    batch_size=256,
    valid_ratio=.2,
   
    max_vocab=999999,
    min_freq=5,
 )
print("|train|=%d" % len(loaders.train_loader.dataset))
print("|valid|=%d" % len(loaders.valid_loader.dataset))

print("|vocab|=%d" % len(loaders.text.vocab))
print("|label|=%d" % len(loaders.label.vocab))

"""

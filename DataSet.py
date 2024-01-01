import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from gensim.models.keyedvectors import KeyedVectors
from transformers import GPT2Tokenizer
import pickle
from tqdm import tqdm
import pandas as pd
import re



class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
        self.tkn2word = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

        # self.label2idx = {}
        # self.idx2label = []

        # 获取 label 的 映射
        # with open(os.path.join(path, "labels.json"), "r", encoding="utf-8") as f:
        #     for line in f:
        #         one_data = json.loads(line)
        #         label, label_desc = one_data["label"], one_data["label_desc"]
        #         self.idx2label.append([label, label_desc])
        #         self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    """
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。

    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    """

    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent
        
        self.train = self.tokenize(os.path.join(path, "ROCStories_train.csv"))
        self.valid = self.tokenize(os.path.join(path, "ROCStories_val.csv"))
        self.test = self.tokenize(os.path.join(path, "ROCStories_test.csv"), True)

        # -----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
        # 先检查本地有没有存好的 embedding_weight，若有则直接读取，若没有则生成并存储
        if os.path.exists("embedding_weight.pkl"):
            with open("embedding_weight.pkl", "rb") as f:
                self.embedding_weight = pickle.load(f)
                print("loaded embedding_weight...")
        else:
            self.embedding_weight = np.zeros((len(self.dictionary.tkn2word), 300))
            self.embedding_weight =None

        # ------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        """
        padding: 将原始的 token 序列补 [PAD] 至预设的最大长度 self.max_token_per_sent
        """
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[: self.max_token_per_sent]
        else:
            return origin_token_seq + [
                "[EOS]" for _ in range(self.max_token_per_sent - len(origin_token_seq))
            ]


    def tokenize(self, path, test_mode=False):
        """
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        """
        idss = []
        labels = []
        data = pd.read_csv(path,encoding='utf-8')
        if test_mode:
            data['story'] = "[BOS] "+data['sentence1']
            data['full'] = "[BOS] "+data['sentence1'] +" "+ data['sentence2'] +" "+ data['sentence3'] +" "+ data['sentence4'] + " "+ data['sentence5']
        else:    
            data['story'] = "[BOS] "+data['sentence1'] +" "+ data['sentence2'] +" "+ data['sentence3'] +" "+ data['sentence4'] + " "+ data['sentence5']
        for i in tqdm(range(len(data['story']))):
                sent = data['story'][i]
                if test_mode:
                    lab = data['full'][i]
                # sent = one_data["sentence"]
                # -----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                # 用空格分词
                # sent = sent.split(" ")
                #用正则表达式分词，保留标点符号,"[BOS]"和"[EOS]"为一个整体保留
                sent = re.findall(r'[\w]+|[.,;!?]|\[\w+\]', sent)
                if test_mode:
                    lab = re.findall(r'[\w]+|[.,;!?]|\[\w+\]', lab)
                    pad_lab = self.pad(lab)
                # print(sent)
                # ------------------------------------------------------end------------------------------------------------------#
                pad_sent = self.pad(sent)
                 # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)
                if test_mode:
                    for word in lab:
                        self.dictionary.add_word(word)
                # 将每一条pad_sent切分为4份，第一份1-21，第二份21-41，第三份41-61，第四份61-81
                if 0:

                    for j in range(4):
                        ids = []
                        label = []
                        is_empty = False
                        for word in pad_sent[j*20:(j+1)*20]:
                            if pad_sent[j*20] == "[EOS]":
                                is_empty = True
                                break
                            ids.append(self.dictionary.add_word(word))
                        if not is_empty:
                            idss.append(ids)
                            # print(ids)
                        is_empty = False
                        for word in pad_sent[j*20+1:(j+1)*20+1]:
                            if pad_sent[j*20] == "[EOS]":
                                is_empty = True
                                break
                            label.append(self.dictionary.add_word(word))
                        if not is_empty:
                            labels.append(label)
                if not test_mode:
                    ids = []
                    label = []
                    for word in pad_sent[0:80]:
                        ids.append(self.dictionary.add_word(word))
                    idss.append(ids)
                    for word in pad_sent[1:81]:
                        label.append(self.dictionary.add_word(word))
                    labels.append(label)
                if test_mode:
                    ids = []
                    label = []
                    for word in pad_sent[0:80]:
                        ids.append(self.dictionary.add_word(word))
                    idss.append(ids)
                    for word in pad_lab[0:80]:
                        label.append(self.dictionary.add_word(word))
                    labels.append(label)
                # else:
                #     ids = []
                #     label = []
                #     for word in pad_sent:
                #         if word == "[PAD]":
                #             break
                #         ids.append(self.dictionary.add_word(word))
                #     idss.append(ids)
                #     # print(ids)
                #     for word in pad_sent:
                #         if word == "[PAD]":
                #             break
                #         label.append(self.dictionary.add_word(word))
                #     labels.append(label)
        idss = torch.tensor(np.array(idss))
        labels = torch.tensor(np.array(labels))#.long()

        return TensorDataset(idss, labels)

class Corpus_GPT(object):
    """
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。

    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    """

    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent
        
        self.train = self.tokenize(os.path.join(path, "ROCStories_train.csv"))
        self.valid = self.tokenize(os.path.join(path, "ROCStories_val.csv"))
        self.test = self.tokenize(os.path.join(path, "ROCStories_test.csv"), True)
    def tokenize(self, path, test_mode=False):
        """
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        """
        idss = []
        labels = []
        masks = []
        data = pd.read_csv(path,encoding='utf-8')
#         special_tokens = {
#     'bos_token': '[BOS]',
# }
        if test_mode:
            data['story'] = data['sentence1']
            data['full'] = data['sentence1'] +" "+ data['sentence2'] +" "+ data['sentence3'] +" "+ data['sentence4'] + " "+ data['sentence5']
        else:    
            data['story'] = data['sentence1'] +" "+ data['sentence2'] +" "+ data['sentence3'] +" "+ data['sentence4'] + " "+ data['sentence5']
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2", cache_dir="./cache")#, **special_tokens)
        # test="[BOS] He is cool."+"<|endoftext|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        for i in tqdm(range(len(data['story']))):
                sent = data['story'][i]
                if test_mode:
                    lab = data['full'][i]
                
                # sent = self.tokenizer.encode(sent,max_length=self.max_token_per_sent,padding="max_length", truncation=True, return_tensors='pt')
                encoding = self.tokenizer(sent, max_length=self.max_token_per_sent, padding="max_length", truncation=True, return_tensors='pt', return_attention_mask=True)
                input_ids = encoding["input_ids"].squeeze().tolist()
                attention_mask = encoding["attention_mask"].squeeze().tolist()
                # print(input_ids)
                idss.append(input_ids)
                if not test_mode:
                    labels.append(input_ids)
                masks.append(attention_mask)
                # print(input_ids,attention_mask)
                # print(sent)
                # sent = self.tokenizer.decode(sent[0],skip_special_tokens=True)
                if test_mode:
                    # lab = self.tokenizer.encode(lab,max_length=self.max_token_per_sent,padding="max_length", truncation=True, return_tensors='pt')
                    label_encoding = self.tokenizer(lab, max_length=self.max_token_per_sent, padding="max_length", truncation=True, return_tensors='pt', return_attention_mask=True)
                    label_input_ids = label_encoding["input_ids"].squeeze().tolist()
                    label_attention_mask = label_encoding["attention_mask"].squeeze().tolist()
                    labels.append(label_input_ids)

                # print("1")
        idss = torch.tensor(np.array(idss))
        labels = torch.tensor(np.array(labels))#.long()
        masks = torch.tensor(np.array(masks))
        print(idss.shape)
        print(labels.shape)
        print(masks.shape)
        return TensorDataset(idss, labels,masks)
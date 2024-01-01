import torch.nn as nn
import torch as torch
import math
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig, PromptTuningConfig, AdaLoraConfig, PromptEncoderConfig
class LSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=128, d_hid=512, nlayers=1, dropout=0.8, embedding_weight=None):
        super(LSTM_model, self).__init__()
        self.ntoken = ntoken
        self.d_emb = d_emb
        self.d_hid = d_hid
        self.nlayers = nlayers
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=False, batch_first=True, dropout=dropout)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 lstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        self.classify = nn.Sequential(
            nn.Linear(d_hid, vocab_size),
            # nn.ReLU(inplace=True),
        )
        #------------------------------------------------------end------------------------------------------------------#
    def forward(self, input, method=-1):
        # print(input.shape)
        self.input_num=input.shape[0]
        x = self.embed(input)
        output,(h_n,c_n) = self.lstm(x)
        # print(output.shape)
        if method == -1:
            print("please choose a method")
            return
        
        #-----------------------------------------------------begin-----------------------------------------------------#
        # method 1: 返回[batch_size, ntoken, vocab_size]的三维张量，每个token对应一个vocab_size维的向量
        if method == 1:
            ans = self.classify(output)
            # print(ans.shape)
            
        #------------------------------------------------------end------------------------------------------------------#
        return ans
    
class GPT2_model(nn.Module):
    def __init__(self, method, ntoken, d_emb=128, d_hid=512, nlayers=1, dropout=0.8, embedding_weight=None):
        super(GPT2_model, self).__init__()
        self.ntoken = ntoken
        self.d_emb = d_emb
        self.d_hid = d_hid
        self.nlayers = nlayers
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
        if method =="zero":
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
        if method =="full":
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            for param in self.gpt.parameters():
                param.requires_grad = True
        if method =="partial_8":
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            for i, layer in enumerate(self.gpt.transformer.h):
                if i < 8:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
        if method =="lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()
        if method =="prefix":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=20)
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()
        if method == "prompt":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=10,
                                            prompt_tuning_init="TEXT",
                                            prompt_tuning_init_text="give one sentence to start the five sentence story:",
                                            tokenizer_name_or_path="gpt2",)
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()
        if method == "adalora":
            peft_config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()
        if method == "p":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=20,
                                            encoder_hidden_size=128)
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()
        if method =="add":
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            for param in self.gpt.parameters():
                param.requires_grad = True
        if method =="step":
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2',return_dict=True)
            for i, layer in enumerate(self.gpt.transformer.h):
                if i < 8:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
    def forward(self, input, labels, attention_mask, method=-1):
        
        if method == -1:
            print("please choose a method")
            return
        if method == "zero":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
            
        if method == "full":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output

        if method == "partial_8":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "lora":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "prefix":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "prompt":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "adalora":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "p":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "add":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        if method == "step":
            output = self.gpt(input, labels=labels,attention_mask=attention_mask)
            ans = output
        return ans
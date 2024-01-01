import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
from DataSet import Corpus_GPT
from Model import GPT2_model
import torch.nn as nn
import os
import time
import pickle
from eval import bleu,rouge,bleu_nltk
def train(is_train,optimizer,scheduler):
    model.train()
    optimizer=optimizer
    scheduler=scheduler
    for epoch in range(num_epochs):
        if is_train:
            if fin_method =="add":
                accumulating_batch_count = 0
            if fin_method =="step":
                if epoch >= 2 and epoch <= 9:
                    layer_num = 7 - (epoch - 2)  # 从第7层开始解冻，到第0层
                    for param in model.gpt.transformer.h[layer_num].parameters():
                        param.requires_grad = True
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr, weight_decay=5e-4)
                    if epoch % 2==0:
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,3,5,8], gamma=0.3)
                    if epoch % 2==1:
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,3,5,8], gamma=0.3)
                    # 统计总参数和可训练参数
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"Total Parameters: {total_params}")
                    print(f"Trainable Parameters: {trainable_params}")
                    with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
                        f.write(f"Total Parameters: {total_params}\n")
                        f.write(f"Trainable Parameters: {trainable_params}\n")
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            for i, data in enumerate(data_loader_train):
                data_in = data[0].to(device)
                target = data[1].to(device)
                masks = data[2].to(device)
                # print(data_in.shape)
                outputs = model(data_in, labels=target,attention_mask=masks, method=fin_method)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                if fin_method =="add":
                    if accumulating_batch_count % 256 == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        model.zero_grad()
                    accumulating_batch_count += 1
                else: 
                    optimizer.step()
                    optimizer.zero_grad()
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(data_loader_train), loss.item()))
        # 每个epoch结束后在验证集上测试
        test_loss = 0
        total = 0
        correct = 0
        
        total_word = 0
        with torch.no_grad():
            for data in tqdm(data_loader_valid, desc='Validation'):
                data_in = data[0].to(device)
                target = data[1].to(device)
                masks = data[2].to(device)
                outputs = model(data_in, labels=target,attention_mask=masks, method=fin_method)
                loss = outputs.loss
                logits = outputs.logits
                prediction = torch.max(logits, dim=-1)[1]
                if fin_method =="prompt":
                    prediction = prediction[:,10:]
                if fin_method =="p":
                    prediction = prediction[:,20:]
                correct += (prediction == target).sum().item()
                total_word += target.size(1)*target.size(0)
                test_loss += loss.item()
                total += 1
                # if fin_method =="full":
                #      scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # scheduler.step()
        # scheduler.step(round(test_loss / total,2))
        # if fin_method != "full":
        if fin_method!="add":
            scheduler.step()
        print(f"Epoch: {epoch+1}, Learning Rate: {current_lr}")
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total_word))
        print('Loss of the network on the test data: {}'.format(test_loss / total))
        #将以上输出保存在log文件中
        with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
            f.write('Loss of the network on the test data: {}\n'.format(test_loss / total))
            f.write('Accuracy of the network on the test data: {} %\n'.format(100 * correct / total_word))
            f.write('Epoch: {}, Learning Rate: {}\n'.format(epoch+1, current_lr))

    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_folder, f'model_gpt_{fin_method}.ckpt'))
    print('model saved to %s' % os.path.join(output_folder, f'model_gpt_{fin_method}.ckpt'))

def predict(steps,method="greedy"):
    model.eval()
    predicts=[]
    ground_truths=[]
    max_times = 10
    
    with open(os.path.join(output_folder, f'predicts_gpt_{fin_method}.txt'), 'a') as f:
        f.write("method:"+method+"\n")
    for i, data in enumerate(data_loader_test):
        data_in = data[0].to(device)
        target = data[1].to(device)
        masks = data[2].to(device)
        # print(data_in.shape)
        #用masks去掉data_in中的pad
        data_in = data_in[masks==1]
        data_in = data_in.unsqueeze(0)
        # print(data_in.shape)
        # print(data_in)
        temperature=1
        if method == "greedy":
            outputs = model.gpt.generate(input_ids=data_in, max_length=steps, num_beams=1,temperature=temperature, no_repeat_ngram_size=2, pad_token_id=dataset.tokenizer.eos_token_id)
        elif method == "top_k_p":
            outputs = model.gpt.generate(input_ids=data_in, max_length=steps, do_sample=True, top_k=50, top_p=0.90,temperature=temperature, no_repeat_ngram_size=2, pad_token_id=dataset.tokenizer.eos_token_id)
        elif method == "beam":
            outputs = model.gpt.generate(input_ids=data_in, max_length=steps, num_beams=10, temperature=temperature ,no_repeat_ngram_size=2, pad_token_id=dataset.tokenizer.eos_token_id)
        out_word = dataset.tokenizer.decode(outputs[0].tolist(),skip_special_tokens=True)
        print("-------------------------------------")
        print(out_word)
        # 将输出保存在predicts.txt中
        with open(os.path.join(output_folder, f'predicts_gpt_{fin_method}.txt'), 'a') as f:
            f.write(out_word+"\n")
        predicts.append(out_word)
        target = dataset.tokenizer.decode(target.tolist()[0],skip_special_tokens=True)
        print(target)
        ground_truths.append(target)
        if i == max_times:
            break
    bleu_score = 0
    rouge_score = 0
    # nltk_bleu_score = 0
    for i in range(len(predicts)):
        # print(type(predicts[i]))
        # print(type(ground_truths[i]))
        bleu_score += bleu(predicts[i],ground_truths[i])
        rouge_score += rouge(predicts[i],ground_truths[i])
        # nltk_bleu_score += bleu_nltk([ground_truths[i]],predicts[i])
    bleu_score /= len(predicts)
    rouge_score /= len(predicts)
    # nltk_bleu_score /= len(predicts)
    print("method:",method)
    print("bleu_score:",bleu_score)
    print("rouge_score:",rouge_score)
    # print("nltk_bleu_score:",nltk_bleu_score)
    #将以上输出保存在log文件中
    with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
        f.write("method:"+method+"\n")
        f.write("bleu_score:"+str(bleu_score)+"\n")
        f.write("rouge_score:"+str(rouge_score)+"\n")
        # f.write("nltk_bleu_score:"+str(nltk_bleu_score)+"\n")
        f.write("-----------------------------------------------------\n")
    return


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('running on device ' + str(device))
    # 以下为超参数，可根据需要修改-----------------------------------------------------
    embedding_dim = 300     # 每个词向量的维度
    batch_size = 16
    # batch_size = 16
    num_epochs =10
    lr = 3e-3
    dataset_folder = 'story_genaration_dataset'
    output_folder = './output'
    max_len = 105 #整个故事最大长度
    is_train = False
    zero_shot = False
    #创建一个list包含所有方法
    fin_method_list = ["step","add","prefix","prompt","p","adalora","full","partial_8","lora","zero"]
    fin_method="full"
#-------------------------------------------------------------------------------
    if os.path.exists("./dataset_GPT.pkl"):
        with open("./dataset_GPT.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = Corpus_GPT(dataset_folder, max_len)
    #将dataset打包为pickle文件，方便下次直接读取
    with open("./dataset_GPT.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print(dataset.vocab_size)
    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=1, shuffle=False)
    for fin_method in fin_method_list:
        # fin_method = "full"
        print("fin_method:",fin_method)
        model = GPT2_model(method=fin_method,ntoken= 105, d_emb=embedding_dim, d_hid=embedding_dim, nlayers=1, dropout=0.8, embedding_weight=None).to(device)                    
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,3,5,8], gamma=0.3)
        if fin_method =="add":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=-1)
        if fin_method =="zero":
            is_train = False
            zero_shot = True
        if is_train:
            with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
                        f.write("  --------------------new_train  "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"------------------------------------------HOHOHO\n")
            train(is_train=True,optimizer=optimizer,scheduler=scheduler)
        elif not zero_shot:
            # 加载保存好的模型
            model.load_state_dict(torch.load(os.path.join(output_folder, f'model_gpt_{fin_method}.ckpt')))
            with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
                        f.write("  --------------------new_test  "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
                        f.write('model loaded from %s\n' % os.path.join(output_folder, f'model_gpt_{fin_method}.ckpt'))
            print('model loaded from %s' % os.path.join(output_folder, f'model_gpt_{fin_method}.ckpt'))
            # train(is_train=False,optimizer=optimizer,scheduler=scheduler)
            pass
        elif zero_shot:
            with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
                        f.write("  --------------------new_test  "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
        # print("predicting")
        predict(max_len,method="greedy")
        predict(max_len,method="top_k_p")
        predict(max_len,method="beam")
        # print(model)


        # 统计总参数和可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        with open(os.path.join(output_folder, f'log_gpt_{fin_method}.txt'), 'a') as f:
            f.write(f"model_structures:{model}\n")
            f.write(f"Total Parameters: {total_params}\n")
            f.write(f"Trainable Parameters: {trainable_params}\n")
            f.write("-----------------------------------------------------\n")

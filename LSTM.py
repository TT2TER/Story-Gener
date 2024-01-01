#使用ROCStorise数据集实现给出第一句话，预测后面四句话的模型
#处理数据
#将数据从.csv中读取出来，切分为定长，向量化
#训练时，使用第1-第20个词预测第2-第21个词
#测试时，使用第1-第10个词预测第11个词，再用第2-第11个词预测第12个词，以此类推
#王晶说不需要mask，用单向LSTM就可以了
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
# import jieba
# from tqdm import tqdm
import re
import os
import random
from DataSet import Corpus
from Model import LSTM_model
import pickle
import time
from transformers import get_linear_schedule_with_warmup
from torch.distributions import Categorical
from eval import bleu,rouge
from tqdm import tqdm
#下面开始写代码
def train(is_train=True):
    model.train()
    for epoch in range(num_epochs):
        if is_train:
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            for i, data in enumerate(data_loader_train):
                # print("input")
                # print(data[0].shape)
                # print(data[1].shape)
                
                #data_in是每个batch的第1-第20个词
                data_in = data[0].to(device)
                #target是每个batch的第2-第21个词
                target = data[1].to(device)
                #tatget=target从第十个词开始算损失
                # target=target[:,10:]
                optimizer.zero_grad()
                output = model(data_in,1)
                # output=output[:,10:,:]
                # print("out&target")
                # print(output.shape)
                # print(target.shape)
                
                loss = loss_function(output.permute(0, 2, 1), target)
                # print(loss.item())
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(data_loader_train), loss.item()))
                    #将以上输出保存在log文件中
                    # with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                    #     f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'.format(epoch + 1, num_epochs, i + 1, len(data_loader_train), loss.item()))
        # 每个epoch结束后在验证集上测试
        test_loss = 0
        correct = 0
        total = 0
        total_word = 0
        with torch.no_grad():
            for data in data_loader_valid:
                data_in = data[0].to(device)
                target = data[1].to(device)
                output = model(data_in,1)
                # print(output.shape)
                loss = loss_function(output.permute(0, 2, 1), target)
                test_loss += loss.item()
                # print(output.shape)
                _, predicted = torch.max(output, 2)
                # print(predicted.shape)
                # print(target.shape)
                total += 1
                # print(target.size(1))
                total_word += target.size(1)*target.size(0)
                correct += (predicted == target).sum().item()
                
        
        current_lr = optimizer.param_groups[0]['lr']
        # scheduler.step()
        scheduler.step(round(test_loss / total,2))
        print(f"Epoch: {epoch+1}, Learning Rate: {current_lr}")
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total_word))
        print('Loss of the network on the test data: {}'.format(test_loss / total))
        #将以上输出保存在log文件中
        with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
            f.write('Loss of the network on the test data: {}\n'.format(test_loss / total))
            f.write('Accuracy of the network on the test data: {} %\n'.format(100 * correct / total_word))
            f.write('Epoch: {}, Learning Rate: {}\n'.format(epoch+1, current_lr))
            
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_folder, 'model.ckpt'))
    print('model saved to %s' % os.path.join(output_folder, 'model.ckpt'))

def t2w(tokens):
    #num/tensor转为word
    return [dataset.dictionary.tkn2word[t] for t in tokens]

def t2n(data_in):
    # 将tensor转为numpy
    data_in = data_in.cpu().numpy()
    data_in = np.squeeze(data_in) #降维
    return data_in


def n2t(data_in):
    # 将numpy转为tensor
    data_in = torch.from_numpy(data_in)
    data_in = torch.unsqueeze(data_in,0)
    return data_in

def del_pad(data_in):
    
    # 将data_in后面的"EOS"去掉
    data_in = data_in.tolist()
    data_in = [x for x in data_in if x != 2]
    #转为np.ndarray
    data_in = np.array(data_in)
    return data_in

def get_gt(gt):
    gt = t2n(gt)
    gt = del_pad(gt)
    gt = t2w(gt)
    return gt

def predict(steps,method="greedy"):
    model.eval()
    predicts=[]
    ground_truths=[]
    max_times = 10
    with torch.no_grad():
        times = 0
        if method == "greedy":
            for data in tqdm(data_loader_test, desc=f'Predicting with {method}'):
                times += 1
                if times > max_times:
                    break
                data_in = data[0]
                data_num = t2n(data_in)
                data_num = del_pad(data_num)
                data_in = n2t(data_num)
                data_word =t2w(data_num)
                # print("第一句:",data_word)
                data_in = data_in.to(device)
                for _ in range(steps):
                    output = model(data_in,1)
                    #只取output第二维最后一个输出
                    output = output[:,-1,:]
                    output = torch.softmax(output,dim=1)
                    value, predicted = torch.max(output, 1)
                    #将预测的词加入data_in中
                    data_in = torch.cat((data_in,predicted.unsqueeze(0)),1)
                    #将预测的词加入data_word中
                    data_word.append(dataset.dictionary.tkn2word[predicted.item()])
                    if predicted.item() == 2:
                        break
                # print("预测结果",data_word)
                predicts.append(data_word)
                gt=get_gt(data[1])
                # print("ground_truth",gt)
                ground_truths.append(gt)
        elif method == "beam":
            beam_width = 10
            for data in tqdm(data_loader_test, desc=f'Predicting with {method}'):
                times += 1
                if times > max_times:
                    break
                data_in = data[0]
                data_num = t2n(data_in)
                data_num = del_pad(data_num)
                data_in = n2t(data_num)
                data_word =t2w(data_num)
                # print("第一句:",data_word)
                beam_list = [(data_num, 1)]
                for _ in range(steps):
                    new_beam_list = []
                    for token, score in beam_list:
                        input = n2t(token)
                        input = input.to(device)
                        output = model(input,1)
                        output = output[:,-1,:]
                        # output = torch.softmax(output,dim=1)
                        #加入temperture_search
                        temp=2
                        output = torch.softmax(output/temp,dim=1)
                        value, slide = torch.topk(output, beam_width)
                        value = value/0.1#减小打分阶数降低的速度
                        # print(value)
                        for i in range(beam_width):
                            new_beam_list.append(
                               (np.append(token, slide.squeeze(0)[i].item()), score*value.squeeze(0)[i].item()))
                    beam_list = sorted(new_beam_list, key=lambda x: x[1], reverse=True)[:beam_width]
                    if beam_list[0][0][-1] == 2:
                        break
                    if len(beam_list[0][0])>steps: break
                out_num = beam_list[0][0]
                data_out = n2t(out_num)
                out_word = t2w(out_num)
                # print("预测结果",out_word)
                predicts.append(out_word)
                gt=get_gt(data[1])
                # print("ground_truth",gt)
                ground_truths.append(gt)

        elif method == "sample":
            for data in tqdm(data_loader_test, desc=f'Predicting with {method}'):
                times += 1
                if times > max_times:
                    break
                data_in = data[0]
                data_num = t2n(data_in)
                data_num = del_pad(data_num)
                data_in = n2t(data_num)
                data_word =t2w(data_num)
                # print("第一句:",data_word)
                data_in = data_in.to(device)
                for _ in range(steps):
                    output = model(data_in,1)
                    #只取output第二维最后一个输出
                    output = output[:,-1,:]
                    output = torch.softmax(output,dim=1)
                    predicted = torch.multinomial(output, 1)
                    #将预测的词加入data_in中
                    data_in = torch.cat((data_in,predicted),1)
                    #将预测的词加入data_word中
                    data_word.append(dataset.dictionary.tkn2word[predicted.item()])
                    if predicted.item() == 2:
                        break
                # print("预测结果",data_word)
                predicts.append(data_word)
                gt=get_gt(data[1])
                # print("ground_truth",gt)
                ground_truths.append(gt)
        elif method == "temp":
            temp = 3
            for data in tqdm(data_loader_test, desc=f'Predicting with {method}'):
                times += 1
                if times > max_times:
                    break
                data_in = data[0]
                data_num = t2n(data_in)
                data_num = del_pad(data_num)
                data_in = n2t(data_num)
                data_word =t2w(data_num)
                # print("第一句:",data_word)
                data_in = data_in.to(device)
                for _ in range(steps):
                    output = model(data_in,1)
                    #只取output第二维最后一个输出
                    output = output[:,-1,:]
                    output = torch.softmax(output,dim=1)
                    dist = Categorical(output/temp)
                    #计算dist的熵
                    # print(dist.entropy())
                    predicted = dist.sample()
                    predicted = predicted.unsqueeze(0)
                    
                    # print(predicted)
                    #将预测的词加入data_in中
                    data_in = torch.cat((data_in,predicted),1)
                    #将预测的词加入data_word中
                    data_word.append(dataset.dictionary.tkn2word[predicted.item()])
                    if predicted.item() == 2:
                        break
                # print("预测结果",data_word)
                predicts.append(data_word)
                gt=get_gt(data[1])
                # print("ground_truth",gt)
                ground_truths.append(gt)
        elif method == "topk":
            k=10
            for data in tqdm(data_loader_test, desc=f'Predicting with {method}'):
                times += 1
                if times > max_times:
                    break
                data_in = data[0]
                data_num = t2n(data_in)
                data_num = del_pad(data_num)
                data_in = n2t(data_num)
                data_word =t2w(data_num)
                # print("第一句:",data_word)
                data_in = data_in.to(device)
                for _ in range(steps):
                    output = model(data_in,1)
                    #只取output第二维最后一个输出
                    output = output[:,-1,:]
                    temp=2
                    output = torch.softmax(output/temp,dim=1)#加入温度让分布更平滑
                    value, slide = torch.topk(output, k)
                    # print(value)
                    predicted = torch.multinomial(value, 1)
                    #将预测的词加入data_in中
                    data_in = torch.cat((data_in,slide.squeeze(0)[predicted]),1)
                    #将预测的词加入data_word中
                    data_word.append(dataset.dictionary.tkn2word[slide.squeeze(0)[predicted].item()])
                    if slide.squeeze(0)[predicted].item() == 2:
                        break
                # print("预测结果",data_word)
                predicts.append(data_word)
                gt=get_gt(data[1])
                # print("ground_truth",gt)
                ground_truths.append(gt)

        elif method == "topp":
            p = 0.9
            
            max_times = 10
            for data in tqdm(data_loader_test, desc=f'Predicting with {method}'):
                times += 1
                if times > max_times:
                    break
                data_in = data[0]
                data_num = t2n(data_in)
                data_num = del_pad(data_num)
                data_in = n2t(data_num)
                data_word = t2w(data_num)
                # print("第一句:", data_word)
                data_in = data_in.to(device)
                for _ in range(steps):
                    output = model(data_in, 1)
                    # 只取output第二维最后一个输出
                    output = output[:, -1, :]
                    
                    sorted_out,indices = torch.sort(output,descending=True)
                    weights = torch.softmax(sorted_out,dim=1).squeeze(0).detach()
                    wh= torch.where(torch.cumsum(weights,dim=0)>=p)[0]
                    max_id = wh[0].item() if len(wh)>0 else len(weights)-1
                    id = torch.multinomial(weights[:max_id+1],num_samples=1)
                    predicted = indices.squeeze(0)[id]
                    # 将预测的词加入data_in中
                    data_in = torch.cat((data_in, predicted.unsqueeze(0)), 1)
                    # 将预测的词加入data_word中
                    data_word.append(dataset.dictionary.tkn2word[predicted.item()])
                    if predicted.item() == 2:
                        break
                # print("预测结果", data_word)
                predicts.append(data_word)
                gt = get_gt(data[1])
                # print("ground_truth", gt)
                ground_truths.append(gt)
        else:
            print("please choose a method")
            return
    #将所有预测结果保存在predicts_{method}.txt中
    with open(os.path.join(output_folder, f'predicts_{method}.txt'), 'w') as f:
        for i in range(len(predicts)):
            f.write(" ".join(predicts[i])+"\n")
    #计算bleu和rouge
    bleu_score = 0
    rouge_score = 0
    rouge_predicts=[]
    rouge_ground_truths=[]
    for i in range(len(predicts)):
        bleu_score += bleu(predicts[i],ground_truths[i])
        #将predicts[i]和ground_truths[i]转为str
        rouge_predicts.append(" ".join(predicts[i]))
        rouge_ground_truths.append(" ".join(ground_truths[i]))
        rouge_score += rouge(rouge_predicts[i],rouge_ground_truths[i])
        # print("predict:",type(rouge_predicts[i]))
        # print("ground_truth:",type(rouge_ground_truths[i]))
        # print(type(predicts[i]))
        # print(type(ground_truths[i]))
        # rouge_score += rouge(predicts[i],ground_truths[i])
    bleu_score /= len(predicts)
    rouge_score /= len(predicts)
    print("method:",method)
    print("bleu_score:",bleu_score)
    print("rouge_score:",rouge_score)
    #将以上输出保存在log文件中
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write("method:"+method+"\n")
        f.write("bleu_score:"+str(bleu_score)+"\n")
        f.write("rouge_score:"+str(rouge_score)+"\n")
        f.write("-----------------------------------------------------\n")
    return




if __name__ == '__main__':     
    # 首先读数据
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('running on device ' + str(device))
# 以下为超参数，可根据需要修改-----------------------------------------------------
    embedding_dim = 300     # 每个词向量的维度
    batch_size = 256
    # batch_size = 16
    num_epochs = 200
    lr = 3e-3
    dataset_folder = 'story_genaration_dataset'
    output_folder = './output'
    max_len = 105 #整个故事最大长度
    is_train = False
#-------------------------------------------------------------------------------
    if os.path.exists("./dataset.pkl"):
        with open("./dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = Corpus(dataset_folder, max_len)
    # print(dataset)
    #将dataset打包为pickle文件，方便下次直接读取
    with open("./dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    # print(type(dataset.train))
    vocab_size = len(dataset.dictionary.tkn2word)

    print("vocab_size:",vocab_size)
    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=1, shuffle=False)
    # vocab_size=28611
    model = LSTM_model(vocab_size=vocab_size, ntoken=20, d_emb=embedding_dim,embedding_weight=dataset.embedding_weight).to(device)
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()#reduction="mean", label_smoothing=0.1)
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,25, 40], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.3)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=90)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    if is_train:
        with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                    f.write("  new_train  "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
        train(is_train=True)
    else:
        #加载保存好的模型
        model.load_state_dict(torch.load(os.path.join(output_folder, 'model6.ckpt')))
        with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                    f.write("  new_test  "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
                    f.write('model loaded from %s\n' % os.path.join(output_folder, 'model6.ckpt'))
        print('model loaded from %s' % os.path.join(output_folder, 'model6.ckpt'))
        # train(is_train=False)
    print("predicting")
    
    
    predict(max_len,method="topp")
    predict(max_len,method="topk")
    predict(max_len,method="beam")
    predict(max_len,method="greedy")
    predict(max_len,method="sample")
    predict(max_len,method="temp")
 

import argparse
from sklearn.metrics import roc_auc_score,r2_score,mean_squared_error
from utils_fp_downstream import *
from torch_geometric.loader import DataLoader
from encoder_gnn import GINNet, GATNet
from model_gnn_fp_downstream import Model_gat_fp, Model_gin_fp, Model_gnn_fp
import torch.nn as nn
import math
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

device = torch.device('cuda')

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # CPU
        torch.cuda.manual_seed(seed) # GPU
        torch.cuda.manual_seed_all(seed) # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def train(model, device, data_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    train_pred = torch.Tensor()
    train_y = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()
    total_loss = 0.0
    loss_values = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, y, w, ew, xw, out_loss, y_loss, w_loss = model(data)
        pred_loss = nn.Sigmoid()(out_loss)
        pred = nn.Sigmoid()(out)
        train_pred = torch.cat((train_pred, torch.Tensor(pred.cpu().data.numpy())), 0)
        train_y = torch.cat((train_y, torch.Tensor(y.cpu().data.numpy())), 0)
        feature_weight = torch.cat((feature_weight, torch.Tensor(xw.cpu().data.numpy())), 0)
        edge_weight = torch.cat((edge_weight, torch.Tensor(ew.cpu().data.numpy())), 0)
        loss = nn.BCELoss(weight=w_loss,reduction='mean')(pred_loss,y_loss)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss/(batch_idx+1)))
    
    avg_loss = total_loss / len(data_loader)
    return train_pred, train_y, feature_weight, edge_weight, avg_loss

def predicting(model, device, data_loader):
    model.eval()     
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()  
    val_loss = 0.0
    val_loss_values = []
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():      
        for data in data_loader:
            data = data.to(device)
            out, y, w, ew, xw, out_loss, y_loss, w_loss = model(data)
            pred = nn.Sigmoid()(out)
            pred_loss = nn.Sigmoid()(out_loss)
            pred = pred.to('cpu')
            y_ = y.to('cpu')
            ew = ew.to('cpu')
            xw = xw.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
            edge_weight = torch.cat((edge_weight, ew), 0)
            feature_weight = torch.cat((feature_weight, xw),0)
            loss = nn.BCELoss(weight=w_loss,reduction='mean')(pred_loss,y_loss)
            val_loss += loss.item()
    avg_loss = val_loss / len(data_loader)

    return total_preds.numpy().flatten(), total_labels.numpy().flatten(), edge_weight.numpy(), feature_weight.numpy(), avg_loss

def caculate_auc(array1,array2,m):
    M = len(array1)
    auc_list = []
    
    for i in range(m):
        sub_array1 = array1[i::m]
        sub_array2 = array2[i::m]
        non_999_indices = sub_array1 != 999
        sub_array1 = sub_array1[non_999_indices]
        sub_array2 = sub_array2[non_999_indices]
        
        if np.unique(sub_array1).size>1:
            auc = roc_auc_score(sub_array1,sub_array2)
            auc_list.append(auc)
           
    return np.mean(auc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DGCL downstream with attention method')
    parser.add_argument('--path', default='down_task')
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--task', default='bace')
    parser.add_argument('--random_seed', default=9, type=int)
    parser.add_argument('--mode', default=0, type=int)

    args = parser.parse_args()     
    print(args)
    
    batch_size, epochs = args.batch_size, args.epochs   
    task, random_seed, mode = args.task, args.random_seed, args.mode
    
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # CPU
        torch.cuda.manual_seed(seed) # GPU
        torch.cuda.manual_seed_all(seed) # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    set_seed(random_seed)

    LOG_INTERVAL = 20
    
    clr_tasks = {'bbbp': 1, 'hiv': 1, 'bace': 1, 'tox21': 12, 'clintox': 2, 'sider': 27, 'MUV': 17, 'toxcast':617, 'PCBA':128}
    task_num = clr_tasks[task]

    train_data = TestbedDataset(root=args.path, dataset='train', task=task)
    valid_data = TestbedDataset(root=args.path, dataset='valid', task=task)
    test_data = TestbedDataset(root=args.path, dataset='test', task=task)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    encoder_file1 = '200_model_encoder_gat_128_200_in-vitro' 
    encoder_file2 = '200_model_encoder_gin_128_200_in-vitro'
    encoder_file = 'encoder_gnn_' + str(batch_size)
  
    
    model_encoder1 = GATNet().cuda()
    model_encoder2 = GINNet().cuda()
    model_encoder1.load_state_dict(torch.load('results/model/'+ encoder_file1 + '.pkl', map_location='cuda:0'))
    model_encoder2.load_state_dict(torch.load('results/model/'+ encoder_file2 + '.pkl', map_location='cuda:0'))
    model = Model_gnn_fp(n_output=task_num, encoder1=model_encoder1,encoder2=model_encoder2).cuda()
    
    if mode == 0:
        for param in model.encoder1.parameters():
            param.requires_grad = False
        for param in model.encoder2.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-7)
    elif mode == 1:
        optimizer = torch.optim.Adam([{'params':model.encoder1.parameters(),'lr':1e-4,'weight_decay':1e-2},{'params':model.encoder2.parameters(),'lr':1e-4,'weight_decay':1e-2},{'params':model.pre.parameters(),'lr':1e-3,'weight_dacay':1e-7}])
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    save_file ='{}'.format(task)
    if not os.path.exists('results/down_task/clr/'+ task +'/'+str(random_seed)+'_'+encoder_file+'_'+task):
        os.makedirs('results/down_task/clr/'+ task +'/'+str(random_seed)+'_'+encoder_file+'_'+task)
    save_name = 'results/down_task/clr/' + task +'/'+ str(random_seed) + '_'+ encoder_file + '_'+task
    result_file_name = save_name+'/'+save_file+'_result.csv'
    valid_AUCs = save_name+'/'+save_file+'_validAUCs.txt'
    test_AUCs = save_name+'/'+save_file+'_testAUCs.txt'
    train_AUCs = save_name+'/'+save_file+'_trainAUCs.txt'
    model_file_name =save_name+'/'+save_file+'_encoder.pkl'
    AUCs = ('Epoch\tAUC')

    with open(valid_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    with open(test_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    with open(train_AUCs, 'w') as f:
        f.write(AUCs + '\n')
 

    best_auc = 0
    stopping_monitor = 0
    train_losses = []
    valid_losses = []
    test_losses = []
    for epoch in range(epochs+1):
        train_pred,train_y,ew,xw,test_loss= train(model, device, train_data_loader, optimizer, epoch + 1)
        train_losses.append(test_loss)
        valid_pred,valid_true,ew,xw,val_loss= predicting(model,device,valid_data_loader)
        valid_losses.append(val_loss)
        test_pred,test_true,ew1,xw1,test_loss = predicting(model,device,test_data_loader)
        test_losses.append(test_loss)
        
        if (epoch + 0) % 5 == 0:
            train_auc = caculate_auc(train_y,train_pred,task_num)
            AUCs = [epoch,train_auc]
            save_AUCs(AUCs,train_AUCs)
            
            print('train_AUC:',train_auc)

            valid_auc = caculate_auc(valid_true,valid_pred,task_num)
            AUCs = [epoch, valid_auc]
            print('valid_AUC: ', AUCs)

            if best_auc < valid_auc:
                best_auc = valid_auc
                stopping_monitor = 0
                print('best_auc：', best_auc)
                save_AUCs(AUCs, valid_AUCs)
                print('save model weights')
                torch.save(model.state_dict(), model_file_name)
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 20:
                break
    
    model.load_state_dict(torch.load(model_file_name))
    test_pred,test_true,ew,xw,loss = predicting(model, device,test_data_loader)
    print('pred value：',test_pred)
    print('true value：',test_true)
    print('loss value：',loss)
    test_pred_value = save_name+'/'+save_file+'_pred.txt'
    test_true_value = save_name+'/'+save_file+'_true.txt'
    save_AUCs(test_pred,test_pred_value)
    save_AUCs(test_true,test_true_value)
    
    test_auc = caculate_auc(test_true,test_pred,task_num)
    AUCs = [0, test_auc]
    print(task,random_seed,'test_AUC: ', AUCs)

    save_AUCs(AUCs, test_AUCs)
    
    plt.figure()
    plt.plot(train_losses,label='Training loss')
    plt.plot(valid_losses,label='Validation loss')
    plt.plot(test_losses,label='Test loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_name+'/'+task+'_loss.png')
    plt.show()
    
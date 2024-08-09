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
from math import sqrt
np.set_printoptions(threshold=np.inf)
import random
import matplotlib.pyplot as plt

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

def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  
        absError.append(abs(val))  
    mae=sum(absError)/len(absError)
    mse=sum(squaredError)/len(squaredError)  
    rmse = mse ** 0.5
    return mae, mse, rmse

def train(model, device, data_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    train_pred = torch.Tensor()
    train_y = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()
    total_loss = 0.0
    for batch_idx, data in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out, y, w, ew, xw, out_loss, y_loss, w_loss = model(data)
        pred = out
        train_pred = torch.cat((train_pred, torch.Tensor(pred.cpu().data.numpy())), 0)
        train_y = torch.cat((train_y, torch.Tensor(y.cpu().data.numpy())), 0)
        feature_weight = torch.cat((feature_weight, torch.Tensor(xw.cpu().data.numpy())), 0)
        edge_weight = torch.cat((edge_weight, torch.Tensor(ew.cpu().data.numpy())), 0)
        loss = loss_fn(pred,y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
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
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():      
        for data in data_loader:
            data = data.to(device)
            out, y, w, ew, xw, out_loss, y_loss, w_loss = model(data)
            pred = out
            pred = pred.to('cpu')
            y_ = y.to('cpu')
            ew = ew.to('cpu')
            xw = xw.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
            edge_weight = torch.cat((edge_weight, ew), 0)
            feature_weight = torch.cat((feature_weight, xw),0)    
    avg_loss = val_loss / len(data_loader)
    return total_preds.numpy().flatten(), total_labels.numpy().flatten(), edge_weight.numpy(), feature_weight.numpy(), avg_loss  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ATMOL downstream')
    parser.add_argument('--path', default='down_task', help='down_task orginal data for input')
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--task', default='lipo', help='Name of downstream task')
    parser.add_argument('--random_seed', default=9, type=int, help='the seed of downstream task')

    args = parser.parse_args()     
    print(args)
    
    batch_size, epochs = args.batch_size, args.epochs 
    task, random_seed = args.task, args.random_seed
    
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
  
      
    train_data = TestbedDataset(root=args.path, dataset='train', task=task)
    valid_data = TestbedDataset(root=args.path, dataset='valid', task=task)
    test_data = TestbedDataset(root=args.path, dataset='test', task=task)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=None)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=None)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=None)
    
    encoder_file1 = '200_model_encoder_gat_128_200_in-vitro' 
    encoder_file2 = '200_model_encoder_gin_128_200_in-vitro'
    encoder_file = '200_model_encoder_gnn_128_200_in-vitro'
  
    
    model_encoder1 = GATNet().cuda()
    model_encoder2 = GINNet().cuda()
    model_encoder1.load_state_dict(torch.load('results/model/'+ encoder_file1 + '.pkl', map_location='cuda:0'))
    model_encoder2.load_state_dict(torch.load('results/model/'+ encoder_file2 + '.pkl', map_location='cuda:0'))
    model = Model_gnn_fp(n_output=1, encoder1=model_encoder1,encoder2=model_encoder2).cuda()
    for param in model.encoder1.parameters():
        param.requires_grad = False
    for param in model.encoder2.parameters():
        param.requires_grad = False
        
        
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay=1e-7)
    

    save_file ='{}'.format(task)
    if not os.path.exists('results/down_task/clr/'+ task+'/'+str(random_seed)+'_'+encoder_file+'_'+task):
        os.makedirs('results/down_task/clr/'+ task+'/'+str(random_seed)+'_'+encoder_file+'_'+task)
    save_name = 'results/down_task/clr/' + task+'/'+ str(random_seed) + '_'+ encoder_file + '_'+task
    result_file_name = save_name+'/'+save_file+'_result.csv'
    train_RMSE = save_name+'/'+save_file+'_trainRMSEs.txt'
    valid_RMSE = save_name+'/'+save_file+'_validRMSEs.txt'
    test_RMSE = save_name+'/'+save_file+'_testRMSEs.txt'
    model_file_name =save_name+'/'+save_file+'_encoder.pkl'
    RMSEs = ('Epoch\tRMSE\trmse\tR2\tMSE\tmse\tmae')

    with open(valid_RMSE, 'w') as f:
        f.write(RMSEs + '\n')
    with open(test_RMSE, 'w') as f:
        f.write(RMSEs + '\n')
    with open(train_RMSE,'w') as f:
        f.write(RMSEs+'\n')

    best_rmse = 10000
    stopping_monitor = 0
    train_losses = []
    valid_losses = []
    test_losses = []
    for epoch in range(epochs+1):
        train_pred,train_y,ew,xw,test_loss = train(model, device, train_data_loader, optimizer, epoch + 1)
        train_losses.append(test_loss)
        valid_pred,valid_true,ew,xw,val_loss= predicting(model,device,valid_data_loader)
        valid_losses.append(val_loss)
        test_pred,test_true,ew1,xw1,test_loss = predicting(model,device,test_data_loader)
        test_losses.append(test_loss)

        if (epoch + 0) % 10 == 0:
            train_mae, train_mse, train_rmse = compute_mae_mse_rmse(train_y,train_pred)
            train_R2 = r2_score(train_y,train_pred)
            train_MSE = mean_squared_error(train_y,train_pred)
            train_RMSE0 = sqrt(train_MSE)
            train_RMSEs = [epoch, train_RMSE0, train_rmse, train_R2, train_MSE, train_mse, train_mae]
            print('train_RMSE:',train_RMSEs)
            save_RMSEs(train_RMSEs,train_RMSE)

            mae, mse, rmse = compute_mae_mse_rmse(valid_true, valid_pred)
            R2 = r2_score(valid_true, valid_pred)
            MSE = mean_squared_error(valid_true, valid_pred)
            RMSE = sqrt(MSE)
            
            RMSEs = [epoch, RMSE, rmse, R2, MSE, mse, mae]
            print('valid_RMSE: ', RMSEs)
            
            if best_rmse > RMSE:
                best_rmse = RMSE
                stopping_monitor = 0
                print('best_rmse：', best_rmse)
                
                save_RMSEs(RMSEs,valid_RMSE)
                print('save model weights')
                torch.save(model.state_dict(), model_file_name)
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 20:
                break
    
    model.load_state_dict(torch.load(model_file_name))
    test_pred, test_true, __, __, __ = predicting(model, device,test_data_loader)
    print('pred value：',test_pred)
    print('true value：',test_true)
    test_pred_value = save_name+'/'+save_file+'_pred.txt'
    test_true_value = save_name+'/'+save_file+'_true.txt'
    save_RMSEs(test_pred,test_pred_value)
    save_RMSEs(test_true,test_true_value)
    
    mae, mse, rmse = compute_mae_mse_rmse(test_true, test_pred)
    R2 = r2_score(test_true, test_pred)
    MSE = mean_squared_error(test_true, test_pred)
    RMSE = sqrt(MSE)
    RMSEs = [0, RMSE, rmse, R2, MSE, mse, mae]
   
    print(task,random_seed,'test_RMSE: ', RMSEs)
 
    save_RMSEs(RMSEs, test_RMSE)
    
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
import argparse
import time
import torch.optim as optim
import torch_geometric
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from utils_gnn_pretrain import *
from model_gnn_pre import GNNCon
from nt_xent import NT_Xent
from encoder_gnn import GATNet,GINNet

device = torch.device('cuda')


def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    feature_graph = torch.Tensor()
    feature_org = torch.Tensor()
    edge_weight = torch.Tensor()
    feature_weight = torch.Tensor()
    for tem in train_bar:
        graph1, out_1, org2, out_2,ew,xw = net(tem)
        feature_graph = torch.cat((feature_graph, torch.Tensor(graph1.cpu().data.numpy())), 0)
        feature_org = torch.cat((feature_org, torch.Tensor(org2.cpu().data.numpy())), 0)
        edge_weight = torch.cat((edge_weight, torch.Tensor(ew.cpu().data.numpy())))
        feature_weight = torch.cat((feature_weight, torch.Tensor(xw.cpu().data.numpy())))
        criterion = NT_Xent(out_1.shape[0], temperature, 1)
        loss = criterion(out_1, out_2)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num, feature_graph, feature_org, edge_weight.numpy(), feature_weight.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGC Train')
    parser.add_argument('--datafile', default='in-vitro')
    parser.add_argument('--path', default='pretrain')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # args parse
    args = parser.parse_args()
    print(args)
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs

    train_data = TestbedDataset(root=args.path, dataset='in-vitro', patt='_gat')

    model_encoder1 = GATNet()
    model_encoder2 = GINNet()
    model = GNNCon(encoder1=model_encoder1, encoder2=model_encoder2)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)


    # training loop
    datafile = 'in-vitro'
    save_name_pre = '{}_{}_{}'.format(batch_size, epochs, datafile)
    if not os.path.exists('results/'+save_name_pre):
        os.mkdir('results/'+save_name_pre)

    for epoch in range(0, epochs + 1):
        start = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loss, features, org, ew,xw = train(model, train_loader, optimizer)

        if epoch in list(range(0, epochs + 1, 10)):
            torch.save(model_encoder1.state_dict(), 'results/model/' + str(epoch) +'_model_encoder_gat_' + save_name_pre +'.pkl')
            torch.save(model_encoder2.state_dict(), 'results/model/' + str(epoch) +'_model_encoder_gin_' + save_name_pre +'.pkl')

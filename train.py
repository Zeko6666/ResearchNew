import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import sys
os.chdir(sys.path[0])
from models import cnn, resnet, res2net, resnext, sk_resnet, resnest, lstm, dilated_conv, depthwise_conv, shufflenet, vit, dcn, channel_attention, spatial_attention, swin
from Daily_and_Sports_Activities.dataproc import DASA
from UniMiB_SHAR.dataproc import UNIMIB
from PAMAP2.dataproc import PAMAP
from UCI_HAR.dataproc import UCI
from USC_HAD.dataproc import USC
from WISDM.dataproc import WISDM
from OPPORTUNITY.dataproc import OPPO

def parse_args():
    parser = argparse.ArgumentParser(description='Train a HAR task')
    parser.add_argument('--dataset', help='select dataset', choices=dataset_dict.keys(), default='unimib')
    parser.add_argument('--model', help='select network', choices=model_dict.keys(), default='cnn')
    parser.add_argument('--savepath', help='directory for saving model and .npy arrays', default='../HAR-datasets')
    parser.add_argument('--batch', type=int, help='batch_size', default=128)
    parser.add_argument('--epoch', type=int, help='epoch', default=100)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0005)
    parser.add_argument('--patience', type=int, help='early stopping patience', default=10)
    parser.add_argument('--save_model_path', help='path to save the best model', default='./best_model.pth')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Dictionaries for datasets and models
    dataset_dict = {
        'uci': UCI, 'unimib': UNIMIB, 'pamap': PAMAP, 
        'usc': USC, 'dasa': DASA, 'wisdm': WISDM, 'oppo': OPPO
    }
    dir_dict = {
        'uci': 'UCI_HAR/UCI HAR Dataset', 'unimib': 'UniMiB_SHAR/UniMiB-SHAR/data', 
        'pamap': 'PAMAP2/PAMAP2_Dataset/Protocol', 'usc': 'USC_HAD/USC-HAD', 
        'dasa': 'Daily_and_Sports_Activities/data', 'wisdm': 'WISDM/WISDM_ar_v1.1', 
        'oppo': 'OPPORTUNITY/OpportunityUCIDataset/dataset'
    }
    model_dict = {
        'cnn':cnn.CNN, 'resnet': resnet.ResNet, 'res2net': res2net.Res2Net, 
        'resnext': resnext.ResNext, 'sknet': sk_resnet.SKResNet, 'resnest': resnest.ResNeSt, 
        'lstm': lstm.LSTM, 'ca': channel_attention.ChannelAttentionNeuralNetwork, 
        'sa': spatial_attention.SpatialAttentionNeuralNetwork, 'dilation': dilated_conv.DilatedConv, 
        'depthwise': depthwise_conv.DepthwiseConv, 'shufflenet': shufflenet.ShuffleNet, 
        'dcn': dcn.DeformableConvolutionalNetwork, 'vit': vit.VisionTransformer, 
        'swin': swin.SwinTransformer
    }
    
    args = parse_args()
    args.savepath = os.path.abspath(args.savepath) if args.savepath else ''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = args.batch
    EP = args.epoch
    LR = args.lr
    patience = args.patience

    # Initialize Early Stopping and Model Saving Variables
    best_val_acc = 0
    patience_counter = 0
    
    print('\n==================================================【HAR 训练任务开始】===================================================\n')
    print(f'Dataset: {args.dataset}\nModel: {args.model}\nEpochs: {EP}\nBatch Size: {BS}\nLearning Rate: {LR}\nDevice: {device}')
        
    '''数据集加载'''
    dataset_name = dir_dict[args.dataset].split('/')[0]
    dataset_saved_path = os.path.join(args.savepath, dataset_name)

    if os.path.exists(dataset_saved_path):
        train_data, test_data, train_label, test_label = np.load(f'{dataset_saved_path}/x_train.npy'), np.load(f'{dataset_saved_path}/x_test.npy'), np.load(f'{dataset_saved_path}/y_train.npy'), np.load(f'{dataset_saved_path}/y_test.npy')
    else:
        train_data, test_data, train_label, test_label = dataset_dict[args.dataset](dataset_dir=dir_dict[args.dataset], SAVE_PATH=args.savepath)

    X_train = torch.from_numpy(train_data).float().unsqueeze(1)
    X_test = torch.from_numpy(test_data).float().unsqueeze(1)
    Y_train = torch.from_numpy(train_label).long()
    Y_test = torch.from_numpy(test_label).long()

    category = len(set(Y_test.tolist()))
    print('x_train_tensor shape:', X_train.shape)
    print('x_test_tensor shape:', X_test.shape)
    print(f'Number of categories: {category}')

    '''模型加载'''
    net = model_dict[args.model](X_train.shape, category).to(device)
    print(net)

    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BS, shuffle=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.001)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, EP//3, 0.5)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    '''训练'''
    for i in range(EP):
        net.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            with autocast():
                out = net(data)
                loss = loss_fn(out, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_sch.step()

        # Validation
        net.eval()
        cor = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                with autocast():
                    out = net(data)
                _, pre = torch.max(out, 1)
                cor += (pre == label).sum()
        val_acc = cor.item() / len(Y_test)
        print(f'epoch: {i}, train-loss: {loss:.4f}, val-acc: {val_acc:.4f}')
        
        # Early stopping and saving best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(net.state_dict(), args.save_model_path)
            print(f'New best model saved with accuracy: {best_val_acc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {i} due to no improvement in validation accuracy for {patience} epochs')
                break

    print(f'Training completed. Best validation accuracy: {best_val_acc:.4f}')

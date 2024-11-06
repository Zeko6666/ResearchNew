import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, roc_curve, precision_recall_curve
import sys
import matplotlib.pyplot as plt
# Change working directory if needed
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
    parser.add_argument('--savepath', help='the dir-path of the .npy array for saving', default='../HAR-datasets')
    parser.add_argument('--batch', type=int, help='batch_size', default=128)
    parser.add_argument('--epoch', type=int, help='epoch', default=100)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0005)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Dataset and model dictionaries
    dataset_dict = {
        'uci': UCI,
        'unimib': UNIMIB,
        'pamap': PAMAP,
        'usc': USC,
        'dasa': DASA,
        'wisdm': WISDM,
        'oppo': OPPO
    }
    dir_dict = {
        'uci': 'UCI_HAR/UCI HAR Dataset',
        'unimib': 'UniMiB_SHAR/UniMiB-SHAR/data',
        'pamap': 'PAMAP2/PAMAP2_Dataset/Protocol',
        'usc': 'USC_HAD/USC-HAD',
        'dasa': 'Daily_and_Sports_Activities/data',
        'wisdm': 'WISDM/WISDM_ar_v1.1',
        'oppo': 'OPPORTUNITY/OpportunityUCIDataset/dataset'
    }
    model_dict = {
        'cnn': cnn.CNN, 
        'resnet': resnet.ResNet,
        'res2net': res2net.Res2Net,
        'resnext': resnext.ResNext,
        'sknet': sk_resnet.SKResNet,
        'resnest': resnest.ResNeSt,
        'lstm': lstm.LSTM,
        'ca': channel_attention.ChannelAttentionNeuralNetwork,
        'sa': spatial_attention.SpatialAttentionNeuralNetwork,
        'dilation': dilated_conv.DilatedConv,
        'depthwise': depthwise_conv.DepthwiseConv,
        'shufflenet': shufflenet.ShuffleNet,
        'dcn': dcn.DeformableConvolutionalNetwork,
        'vit': vit.VisionTransformer,
        'swin': swin.SwinTransformer
    }

    args = parse_args()
    args.savepath = os.path.abspath(args.savepath) if args.savepath else ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = args.batch
    EP = args.epoch
    LR = args.lr

    print(f'\nStarting HAR Training Task\nDataset: {args.dataset}\nModel: {args.model}\nPath: {args.savepath}\nEpochs: {EP}\nBatch Size: {BS}\nLearning Rate: {LR}\nDevice: {device}\n')

    # Load dataset
    dataset_name = dir_dict[args.dataset].split('/')[0]
    dataset_saved_path = os.path.join(args.savepath, dataset_name)

    # Load or preprocess data
    if os.path.exists(dataset_saved_path):
        print(f'Dataset {dataset_name} exists in {dataset_saved_path}, loading...')
        train_data, test_data, train_label, test_label = np.load(f'{dataset_saved_path}/x_train.npy'), np.load(f'{dataset_saved_path}/x_test.npy'), np.load(f'{dataset_saved_path}/y_train.npy'), np.load(f'{dataset_saved_path}/y_test.npy')
    else:
        train_data, test_data, train_label, test_label = dataset_dict[args.dataset](dataset_dir=dir_dict[args.dataset], SAVE_PATH=args.savepath)

    # Convert to tensors
    X_train = torch.from_numpy(train_data).float().unsqueeze(1)
    X_test = torch.from_numpy(test_data).float().unsqueeze(1)
    Y_train = torch.from_numpy(train_label).long()
    Y_test = torch.from_numpy(test_label).long()

    category = len(set(Y_test.tolist()))
    print(f'Tensor Shapes:\nx_train_tensor: {X_train.shape}\nx_test_tensor: {X_test.shape}\nCategories: {category}')

    # Initialize model
    net = model_dict[args.model](X_train.shape, category).to(device)
    print(net)

    # Create data loaders
    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BS, shuffle=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.001)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, EP // 3, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training loop
    print('\nStarting Training\n')
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

        # Evaluation on test set
        net.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                with autocast():
                    out = net(data)
                _, preds = torch.max(out, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        mcc = matthews_corrcoef(all_labels, all_preds)
        cohen_kappa = cohen_kappa_score(all_labels, all_preds)
        val_accuracies , train_losses = [] , []
         # Store validation accuracy for learning curve
        val_accuracies.append(accuracy)
        # For AUC, ensure one-hot encoding for multiclass if needed
        try:
            auc = roc_auc_score(np.eye(category)[all_labels], np.eye(category)[all_preds], multi_class='ovr')
        except ValueError:
            auc = float('nan')  # If AUC isn't calculable in multiclass setup
        
        print(f'Epoch: {i+1}/{EP}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}, Cohen\'s Kappa: {cohen_kappa:.4f}')
         
         # ROC and Precision-Recall Curves for the final epoch
        if i == EP - 1:
            for class_idx in range(category):
                # Get binary labels for the current class
                binary_labels = np.array([1 if y == class_idx else 0 for y in all_labels])
                binary_preds = np.array([1 if p == class_idx else 0 for p in all_preds])
                
                # ROC curve
                fpr, tpr, _ = roc_curve(binary_labels, binary_preds)
                plt.plot(fpr, tpr, label=f'Class {class_idx}')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (Final Epoch)')
            plt.legend()
            plt.show()

            # Precision-Recall curve
            for class_idx in range(category):
                binary_labels = np.array([1 if y == class_idx else 0 for y in all_labels])
                binary_preds = np.array([1 if p == class_idx else 0 for p in all_preds])
                precision, recall, _ = precision_recall_curve(binary_labels, binary_preds)
                plt.plot(recall, precision, label=f'Class {class_idx}')
                
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve (Final Epoch)')
            plt.legend()
            plt.show()

    # Plot learning curve
    plt.plot(range(1, EP + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EP + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
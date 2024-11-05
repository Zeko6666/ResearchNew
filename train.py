import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, 
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
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

def plot_curves(train_losses, val_accuracies):
    # Plotting learning curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()

def evaluate_model(net, test_loader, device):
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            with autocast():
                out = net(data)
            _, predicted = torch.max(out, 1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(out.softmax(dim=1)[:, 1].cpu().numpy())  # Probabilities for ROC

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'Cohen Kappa: {kappa:.4f}')
    print('Confusion Matrix:\n', conf_matrix)

    # Plot ROC and Precision-Recall Curves
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

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
    train_losses = []
    val_accuracies = []
    
    # Load data, model, etc. - similar to previous code
    
    for i in range(EP):
        # Training loop (same as before)
        train_losses.append(loss.item())
        
        # Validation accuracy
        val_acc = cor.item() / len(Y_test)
        val_accuracies.append(val_acc)
        
        # Early stopping and model saving logic (same as before)
    
    # Plot learning curve
    plot_curves(train_losses, val_accuracies)

    # Evaluate on the test set
    evaluate_model(net, test_loader, device)
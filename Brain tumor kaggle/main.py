from cnn_model import Net
from data_model import BrainMRI
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler 
import argparse
import os
import utils

def load_data(path = '.\data', train_ratio = 0.75):

    data = BrainMRI(path)
    train_length = (int)(train_ratio*data.n_samples)
    test_length = data.n_samples - train_length
    train, test = torch.utils.data.random_split(data.data, [train_length, test_length])
    # trainset = DataLoader(train, batch_size = 20, shuffle = True)
    # testset = DataLoader(test, batch_size = 20, shuffle = True)
    return train, test

def evaluate(model, eval_dataset,criterion, opts):
    
    model.eval()   
    vrunning_loss = 0.0
    vrunning_corrects = 0
    num_samples = 0

    for X, y in eval_dataset:
        X = X.to(opts.device)
        y = y.to(opts.device)
        X = X.float()
            
        with torch.no_grad():
            output = model(X.view(-1, 1, opts.IMG_RESIZE,opts.IMG_RESIZE))
            _, preds = torch.max(output, 1)
            loss = criterion(output, y)

        vrunning_loss += loss.item() * X.size(0)
        vrunning_corrects += (preds == y).sum()
        num_samples += preds.size(0)
        vepoch_loss = vrunning_loss/num_samples
        vepoch_acc = (vrunning_corrects.double() * 100)/num_samples

    print('Validation - Loss: {:.4f}, Acc: {:.4f}'.format(vepoch_loss, vepoch_acc))

    return vepoch_loss

def checkpoint(model, opts):

    with open(os.path.join(opts.checkpoint_path, 'model.pt'), 'wb') as f:
        torch.save(model, f)

def save_loss_plot(train_losses, val_losses, opts):
    """Saves a plot of the training and validation loss curves.
    """
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title('BS={}, IMG_RESIZE={}, Std={}'.format(opts.batch_size, opts.IMG_RESIZE,torch.std(val_losses)), fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.checkpoint_path, 'loss_plot-' + str(opts.fold) + '.pdf'))
    plt.close()

def training_loop(model, optimizer, criterion, train_loader,valid_loader, best_val_loss, opts):
        
        train_epoch_loss = []
        val_epoch_loss = []

        for epoch in range(opts.nepochs):
            
            running_loss = 0.0
            running_corrects = 0
            trunning_corrects = 0

            model.train()
            for X,y in train_loader:
            
                X = X.to(opts.device)
                y = y.to(opts.device)
                X = X.float()
                model.zero_grad()

                with torch.set_grad_enabled(True):
                    output = model(X.view(-1, 1, opts.IMG_RESIZE,opts.IMG_RESIZE))
                    _, predictions = torch.max(output,1)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                

                running_loss += loss.item() * X.size(0)
                running_corrects += (predictions == y).sum()
                trunning_corrects += predictions.size(0)

            epoch_loss = running_loss / trunning_corrects
            epoch_acc = (running_corrects.double()*100) / trunning_corrects
            print('\t\t Epoch({}) \n Training: - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            
            val_loss = evaluate(model, valid_loader,criterion, opts)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint(model, opts)

            train_epoch_loss.append(epoch_loss)
            val_epoch_loss.append(val_loss)
        
        save_loss_plot(train_epoch_loss, val_epoch_loss, opts)
        return model, optimizer, best_val_loss

def KFold_Validation(trainset, model, optimizer, criterion, num_splits = 5):
    
    best_val_loss = 1e6
    splits = KFold(n_splits = num_splits, shuffle = True, random_state = 42)
    for fold, (train_idx, valid_idx) in enumerate(splits.split(trainset)):
        print('Fold : {}'.format(fold))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        opts.__dict__['fold'] = fold
        train_loader = torch.utils.data.DataLoader(trainset, sampler= train_sampler, batch_size = opts.batch_size)
        valid_loader = torch.utils.data.DataLoader(trainset, sampler= valid_sampler, batch_size = opts.batch_size)
        model, optimizer, best_val_loss = training_loop(model, optimizer, criterion, train_loader, valid_loader, best_val_loss, opts)

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--nepochs', type=int, default=5,
                        help='The max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The number of examples in a batch.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate (default 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Set the learning rate decay factor.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Set the directry to store the best model checkpoints.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Choose whether to use GPU.')

    return parser


def main(opts):
    
    trainset, testset = load_data('.\data')
    model = Net()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    KFold_Validation(trainset, model, optimizer, criterion)

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()
    opts.__dict__['IMG_RESIZE'] = 150
    opts.__dict__['device'] = 'cpu'

    print_opts(opts)

    model_name = 'rs{}-bs{}'.format(opts.IMG_RESIZE,opts.batch_size)
    opts.checkpoint_path = os.path.join(opts.checkpoint_dir, model_name)

    utils.create_dir_if_not_exists(opts.checkpoint_path)

    main(opts)

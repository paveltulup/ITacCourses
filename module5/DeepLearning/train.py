import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import sys
from pathlib import Path
import argparse

from model import initialize_model
from test import get_metrics

from torch.utils.tensorboard import SummaryWriter


import warnings
warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def log_tensorboard(writer,loss,group,epoch,metrics=None):
    
    writer.add_scalar(f'Loss/{group}', loss,epoch)
    # print(metrics)

    if metrics is not None:
        p,r,f1,accuracy, roc_auc = metrics
        writer.add_scalar(f'Precision/{group}',p,epoch)
        writer.add_scalar(f'Recall/{group}',r,epoch)
        writer.add_scalar(f'F1/{group}',f1,epoch)
        writer.add_scalar(f'Accuracy/{group}',accuracy,epoch)
        writer.add_scalar(f'RocAuc/{group}',roc_auc,epoch)

def create_writer(base_path,prev_run=False):
    path = Path(base_path)  
    exps = [int(p.name[3:]) for p in path.iterdir()]
    inc = 0 if prev_run else 1
    max_exp = max(exps) + inc if exps else 0
    writer = SummaryWriter(f'{base_path}/exp{max_exp}')
    return writer        


def train(model, dataloaders, criterion, optimizer, scheduler, save_to, num_epochs=20, is_inception=False):

    writer = create_writer('runs/')
    cuda = device.type != 'cpu'
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        writer.add_scalar(f'LR', scheduler.get_last_lr()[0],epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            metrics = []
            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            
            # Iterate over data.
            for batch_idx, (inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                # model.zero_grad(set_to_none=True)

                # forward
                with amp.autocast(enabled=cuda):

                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)

                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                # backward
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    # model.zero_grad(set_to_none=True)
                else:
                    metrics.append(get_metrics(labels,outputs))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # if phase == 'val':
                #     print(loss.item())
                #     print(running_loss)
                
                running_corrects += torch.sum(preds == labels.data)
              
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                metrics = np.mean(metrics,axis=0)
                p, r, f1, acc, roc_auc = metrics
                print('Precision: {:.3f} Recall: {:.3f} F1: {:.3f} Accuracy: {:.3f} RocAuc:{:.3f}'.format(p, r, f1, acc, roc_auc))
                log_tensorboard(writer,epoch_loss,'val',epoch, metrics=metrics)
                val_acc_history.append(epoch_acc)
        
            # tb logging
            if phase == 'train':
                log_tensorboard(writer,epoch_loss,'train',epoch)

            # deep copy the model
            if phase == 'val' and roc_auc > best_metric:
                best_metric = roc_auc
                save_model(model, save_to)
            
        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val RocAuc: {:4f}'.format(best_metric))

    return model, val_acc_history


def save_model(model, pth='model.pt'):
    torch.save(model.state_dict(), pth)
    print(f'Successfully saved model to {pth}')


def train_val(opt):

    num_classes = 2
    batch_size = opt.batch_size
    num_epochs = opt.epochs
    lr = opt.lr

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Initialize the model for this run
    model_ft, input_size = initialize_model(opt.model, num_classes, feature_extract, use_pretrained=opt.pretrained)

    # Print the model we just instantiated
    # print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=(15, 70), translate=(0.1, 0.4),scale=(0.5,1.15)),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            # transforms.RandomInvert(),
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.datapath, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'num workers = {nw}')
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=nw, persistent_workers=True) for x in ['train', 'val']}

    # Detect if we have a GPU available
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    if opt.optimizer == 'SGD':
        optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.98, weight_decay=5e-4)
    elif opt.optimizer == 'Adam':
        optimizer_ft = optim.Adam(params_to_update, lr=lr, weight_decay=5e-4, betas=(0.98,0.999))
    elif opt.optimizer == 'AdamW':
        optimizer_ft = optim.AdamW(params_to_update, lr=lr, weight_decay=5e-4, betas=(0.98,0.999), amsgrad=True)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler=scheduler, save_to=opt.save_to, num_epochs=num_epochs, is_inception=(opt.model == "inception"))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default=ROOT / 'dataset', help='dataset path')
    parser.add_argument('--save-to', type=str, default=ROOT / 'model.pt', help='chackpoint save path')
    parser.add_argument('--model', type=str, choices=["resnet", "alexnet", "vgg", "squeezenet", "inception"], default='resnet', help='Models to choose')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--feature-extract', type=bool, default=True)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    train_val(opt)
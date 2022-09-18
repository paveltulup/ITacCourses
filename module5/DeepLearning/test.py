import torch
import torch.nn as nn
from torch.cuda import amp
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score
import numpy as np
import argparse

from model import initialize_model


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_metrics(actual,pred):
    pred = pred.argmax(dim=1).cpu().detach().numpy()
    actual = actual.cpu().detach().numpy()
    
    p = precision_score(actual, pred, average="weighted")
    r = recall_score(actual, pred, average="weighted")
    f1 = f1_score(actual, pred, average="weighted")
    accuracy = accuracy_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)

    return p, r, f1, accuracy, roc_auc


def test(opt):

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = device.type != 'cpu'

    # model_name = "resnet"
    # model, input_size = initialize_model(model_name, num_classes=2, feature_extract=False, use_pretrained=True)
    # input_size = 512

    # is_inception = False
    # model = models.resnet34(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    # input_size = 512

    model, input_size = initialize_model(opt.model, 2, False, False)

    model_path = opt.weights
    cpkt = torch.load(model_path)
    model.load_state_dict(cpkt)
    model = model.to(device)
    model.eval()
    
    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(opt.datapath, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    running_loss = 0.0
    running_corrects = 0
    # outputs = torch.tensor([]).to(device)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    model.zero_grad(set_to_none=True)
    metrics = []
    # Iterate over data.
    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with amp.autocast(enabled=cuda):
            outputs = model(inputs)
            metrics.append(get_metrics(labels, outputs))

    metrics = np.mean(metrics,axis=0)
    # metrics = get_metrics(labels, outputs)
    p, r, f1, acc, roc_auc = metrics
    print('Precision: {:.3f} Recall: {:.3f} F1: {:.3f} Accuracy: {:.3f} RocAuc:{:.3f}'.format(p, r, f1, acc, roc_auc))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default=ROOT / 'dataset/val', help='test dataset path')
    parser.add_argument('--weights', type=str, default=ROOT / 'model.pt', help='path to weights file')
    parser.add_argument('--model', type=str, choices=["resnet", "alexnet", "vgg", "squeezenet", "inception"], default='resnet', help='Models to choose')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    test(opt)
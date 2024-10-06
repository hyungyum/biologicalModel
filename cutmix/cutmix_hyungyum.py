import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

from models.resnet import *

import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler


# CutMix helper function
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, targets, shuffled_targets, lam


# Random bounding box function for CutMix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def inverse_normalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        num_cor = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            num_cor.append(correct_k.clone())
            acc.append(correct_k.mul_(1 / batch_size))
    return acc, num_cor

# Training loop with CutMix applied
def train(epochs, alpha=1.0):
    best_acc = 0.0
    print('[*] Start training')

    scaler = GradScaler('cuda')  # Mixed Precision scaler

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_loss = 0.0
        total_acc = 0.0

        for step, (data, targets) in enumerate(trainloader):
            data, targets = data.to(device, dtype=torch.float), targets.to(device)

            # Apply CutMix augmentation
            data, targets, shuffled_targets, lam = cutmix(data, targets, alpha)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(data)
                # Calculate CutMix loss
                loss = lam * nn.CrossEntropyLoss()(outputs, targets) + (1 - lam) * nn.CrossEntropyLoss()(outputs,
                                                                                                         shuffled_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            total_loss += loss.item()
            acc, _ = accuracy(outputs, targets)
            running_acc += acc[0].item()
            total_acc += acc[0].item()

            if step % 20 == 0:
                avg_loss = running_loss / (step + 1)
                avg_acc = running_acc / (step + 1)
                print(
                    f'[Epoch {epoch}/{epochs}, Step {step}/{len(trainloader)}] Loss {avg_loss:.4f}, Accuracy {avg_acc:.4f}')
                writer.add_scalar('Loss/train_20step', avg_loss, epoch * len(trainloader) + step)
                writer.add_scalar('Accuracy/train_20step', avg_acc, epoch * len(trainloader) + step)

        avg_epoch_loss = total_loss / len(trainloader)
        avg_epoch_acc = total_acc / len(trainloader)
        print(f'[Epoch {epoch}/{epochs}] Epoch Loss: {avg_epoch_loss:.4f}, Epoch Accuracy: {avg_epoch_acc:.4f}')
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', avg_epoch_acc, epoch)
        scheduler.step()
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Model evaluation
        model.eval()
        total_cor = 0
        total_samples = 0
        total_test_loss = 0.0

        with torch.no_grad():
            for step, (data, targets) in enumerate(testloader):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_test_loss += loss.item()
                _, num_cor = accuracy(outputs, targets)
                total_samples += data.size(0)
                total_cor += num_cor[0].item()

        avg_test_loss = total_test_loss / len(testloader)
        acc = total_cor / total_samples
        print(f'Epoch {epoch} : Test Accuracy {acc:.4f}, Test Loss {avg_test_loss:.4f}')
        writer.add_scalar('Accuracy/test_epoch', acc, epoch)
        writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)

        # Save best model
        if acc > best_acc:
            print('[*] Saving model...')
            state = {'model': model.state_dict(), 'acc': acc, 'epoch': epoch}
            if not os.path.isdir('ckpt_cutmix'):
                os.mkdir('ckpt_cutmix')
            torch.save(state, f'ckpt_cutmix/model_{epoch:03d}_{acc:.4f}.pth')
            best_acc = acc

    print(f'Best Test Accuracy: {best_acc:.4f}')


# Test function remains the same
def test(ckpt_path):
    print(f'[*] load {ckpt_path}')
    model.eval()
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'], strict=True)

    total_cor = 0
    total_samples = 0
    with torch.no_grad():
        for step, (data, targets) in enumerate(testloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            outputs = model(data)
            _, num_cor = accuracy(outputs, targets)
            num_cor = num_cor[0].item()
            total_samples += data.size(0)
            total_cor += num_cor
        acc = total_cor / total_samples
        print(f'Test Accuracy {acc:.4f} of Loaded Model {model.__class__.__name__}')

        # TensorBoard logging
        writer.add_scalar('Test Accuracy', acc)

        # Visualize a few test images with predictions
        images = []
        pred_classes = []
        labels = []
        pred = outputs.topk(1, dim=1, largest=True, sorted=True)
        fig, axes = plt.subplots(3, 3, figsize=(15, 5))  # 3 rows, 3 columns
        axes = axes.flatten()
        for k in range(9):  # show the first 9 images
            images.append(
                inverse_normalize(data[k, :, :, :]).detach().cpu().permute(1, 2, 0).numpy())
            pred_classes.append(classes[pred[1][k].item()])
            labels.append(classes[targets[k].item()])
        for k, image in enumerate(images):
            axes[k].imshow(image)
            axes[k].axis('off')
            axes[k].set_title(f'label: {labels[k]}, pred: {pred_classes[k]}', fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # GPU performance optimization
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # TensorBoard writer
    experiment_name = 'resnet18_cutmix_epoch50_20interval'
    log_dir = f'./runs/{experiment_name}'
    writer = SummaryWriter(log_dir=log_dir)

    # Data preparation with normalization only
    print('[*] preparing data')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model setup
    print('[*] building model')
    model = ResNet18()
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and scheduler
    epochs = 50
    params = model.parameters()
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    train(epochs)

    # # Test
    # directory = './ckpt_cutmix'
    # ckpt_list = os.listdir(directory)
    # ckpt_list = [f for f in ckpt_list if os.path.isfile(os.path.join(directory, f)) and model.__class__.__name__ in f]
    # ckpt_list.sort()
    # ckpt_path = os.path.join(directory, ckpt_list[-1])
    # print(ckpt_path)
    # test(ckpt_path=ckpt_path)

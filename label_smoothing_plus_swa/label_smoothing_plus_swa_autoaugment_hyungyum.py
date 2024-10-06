
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from models.resnet import *

import os
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Label Smoothing Loss 정의
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        target = F.one_hot(target, num_classes=pred.size(-1)).float()
        target = target * (1 - self.smoothing) + self.smoothing / pred.size(-1)
        loss = -target * log_probs
        loss = loss.sum(dim=-1).mean()
        return loss

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

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

def inverse_normalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class ToyNetwork(nn.Module):
    def __init__(self):
        super(ToyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc_code = nn.Linear(in_features=128, out_features=128)
        self.fc_output = nn.Linear(128, 10)
        self.apply(initialize_weights)

    def forward(self, x):
        feature_map = self.conv1(x)
        activated = F.relu(feature_map)
        compressed = F.max_pool2d(activated, kernel_size=(2, 2), stride=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(compressed)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        code = self.fc_code(x)
        output = self.fc_output(code)
        return output

def train(epochs):
    best_acc = 0.0
    print('[*] start training')

    swa_start = 35
    swa_model = AveragedModel(model)  # SWA 모델 초기화
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)  # SWA LR 설정
    scaler = GradScaler('cuda')  # Mixed Precision 스케일러

    for epoch in range(1, epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_loss = 0.0
        total_acc = 0.0
        for step, (data, targets) in enumerate(trainloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):  # Mixed precision 적용
                outputs = model(data)
                loss = criterion(outputs, targets)

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
                print(f'[Epoch {epoch}/{epochs}, Step {step}/{len(trainloader)}] Loss {avg_loss:.4f}, Accuracy {avg_acc:.4f}')

                writer.add_scalar('Loss/train_20step', avg_loss, epoch * len(trainloader) + step)
                writer.add_scalar('Accuracy/train_20step', avg_acc, epoch * len(trainloader) + step)

        avg_epoch_loss = total_loss / len(trainloader)
        avg_epoch_acc = total_acc / len(trainloader)
        print(f'[Epoch {epoch}/{epochs}] Epoch Loss: {avg_epoch_loss:.4f}, Epoch Accuracy: {avg_epoch_acc:.4f}')

        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', avg_epoch_acc, epoch)

        # SWA 적용 시점 확인
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()  # SWA 학습률 적용
            print(f'SWA Model updated at epoch {epoch}')
        else:
            scheduler.step()
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        total_cor = 0
        total_samples = 0
        total_test_loss = 0.0

        with torch.no_grad():
            for step, (data, targets) in enumerate(testloader):
                data = data.to(device, dtype=torch.float)
                targets = targets.to(device)
                outputs = model(data)

                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

                _, num_cor = accuracy(outputs, targets)
                num_cor = num_cor[0].item()
                total_samples += data.size(0)
                total_cor += num_cor

            avg_test_loss = total_test_loss / len(testloader)
            acc = total_cor / total_samples
            print(f'Epoch {epoch} : Test Accuracy {acc:.4f}, Test Loss {avg_test_loss:.4f}')

            writer.add_scalar('Accuracy/test_epoch', acc, epoch)
            writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)

        if acc > best_acc:
            print('[*] model saving...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('ckpt_swa_label_smooting'):
                os.mkdir('ckpt_swa_label_smooting')

            path = f'ckpt_swa_label_smooting/model_{model.__class__.__name__}_state_{epoch:03d}_{acc:.4f}.st'
            torch.save(state, path)
            best_acc = acc
    print(f'Best Test Accuracy {best_acc:.4f}')

    # SWA 마지막 처리 (BN 업데이트)
    print('[*] Updating BN statistics for SWA model...')
    swa_model.to(device)  # SWA 모델을 GPU로 이동시킴

    update_bn(trainloader, swa_model, device)

    # SWA 모델 저장
    torch.save(swa_model.state_dict(), './ckpt_swa_label_smooting/swa_model.pth')
    print('SWA model saved successfully!')

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

        writer.add_scalar('Test Accuracy', acc)

        images = []
        pred_classes = []
        labels = []
        pred = outputs.topk(1, dim=1, largest=True, sorted=True)
        fig, axes = plt.subplots(3, 3, figsize=(15, 5))
        axes = axes.flatten()
        for k in range(9):
            images.append(inverse_normalize(data[k, :, :, :]).detach().cpu().permute(1, 2, 0).numpy())
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

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # 최적화된 GPU 성능을 위한 설정

    # TensorBoard writer 설정 (고유한 실험명 사용)
    experiment_name = 'resnet18_label_smoothing_epoch50_20interval_autoaugment'  # 실험명을 설정
    log_dir = f'./runs/{experiment_name}'  # 고유한 로그 디렉토리 이름 생성
    writer = SummaryWriter(log_dir=log_dir)  # 해당 디렉토리에 기록

    # Data preparation
    print('[*] preparing data')
    transform_train = transforms.Compose([
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # AutoAugment 적용
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
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model building
    print('[*] building model')
    model = ResNet18()
    model.to(device)

    # Label Smoothing CrossEntropyLoss 적용
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Optimizer 설정
    epochs = 50
    params = model.parameters()
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # # Train the model
    train(epochs)

    # # Test and save model
    # directory = './ckpt_label_smoothing'
    # ckpt_list = os.listdir(directory)
    # ckpt_list = [f for f in ckpt_list if os.path.isfile(os.path.join(directory, f)) and model.__class__.__name__ in f]
    # ckpt_list.sort()
    # ckpt_path = os.path.join(directory, ckpt_list[-1])
    # print(ckpt_path)
    # test(ckpt_path=ckpt_path)


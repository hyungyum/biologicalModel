import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.optim.swa_utils import AveragedModel, SWALR
from models.resnet import *

import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 사용

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

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

def train(epochs, writer):
    best_acc = 0.0
    print('[*] start training')

    # SWA 모델 생성 (AveragedModel 사용)
    swa_model = AveragedModel(model)
    swa_model.to(device)
    swa_start_epoch = 2 # 30 에폭 이후 SWA 시작
    swa_scheduler = SWALR(optimizer, swa_lr=1e-3) #SWA 학습 스케쥴러

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, (data, targets) in enumerate(trainloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = nn.CrossEntropyLoss(reduction='mean')(outputs, targets)

            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            acc, _ = accuracy(outputs, targets)
            acc_value = acc[0].item()

            running_loss += loss_value
            total_correct += acc_value * data.size(0)
            total_samples += data.size(0)

            if step % 10 == 0:
                print(f'[Epoch {epoch}/{epochs}, Step {step}/{len(trainloader)}] Loss {loss_value:.4f}, Accuracy {acc_value:.4f}')

        # Scheduler 업데이트(SWA 적용 전까지는 일반 스케쥴러 실행)
        if epoch < swa_start_epoch:
            scheduler.step()
        else:
            swa_scheduler.step()

        # Epoch 결과 기록
        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct / total_samples
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # SWA 모델 업데이트 (30 에폭 이후부터)
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            print(f'SWA Model updated at epoch {epoch}')

        # 검증
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for step, (data, targets) in enumerate(testloader):
                data = data.to(device, dtype=torch.float)
                targets = targets.to(device)
                outputs = model(data)
                _, num_cor = accuracy(outputs, targets)
                num_cor = num_cor[0].item()
                total_samples += data.size(0)
                total_correct += num_cor
            acc = total_correct / total_samples
            print(f'Epoch {epoch} : Test Accuracy {acc:.4f}')

        writer.add_scalar('Accuracy/test', acc, epoch)

        if acc > best_acc:
            print('[*] model saving...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('ckpt_0'):
                os.mkdir('ckpt_0')

            path = f'ckpt_0/model_SWA_{model.__class__.__name__}_state_{epoch:03d}_{acc:.4f}.st'
            torch.save(state, path)
            best_acc = acc

    print(f'Best Test Accuracy {best_acc:.4f}')

    # BN 업데이트
    update_bn(trainloader, swa_model,device)

    # SWA 모델 저장
    print('[*] SWA model saving...')
    swa_state = {
        'model': swa_model.state_dict(),
        'acc': best_acc,
    }
    torch.save(swa_state, 'ckpt_0/swa_model.st')

    print('Training Complete with SWA')


def test(ckpt_path):
    print(f'[*] {ckpt_path} 로드 중')
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
        print(f'로드한 모델 {model.__class__.__name__}의 테스트 정확도 {acc:.4f}')

        # 시각화
        images = []
        pred_classes = []
        labels = []
        pred = outputs.topk(1, dim=1, largest=True, sorted=True)
        fig, axes = plt.subplots(3, 3, figsize=(15, 5))  # 3행 3열
        axes = axes.flatten()
        for k in range(9):  # 처음 9개 이미지만 확인
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

    # TensorBoard SummaryWriter 생성
    writer = SummaryWriter('runs/SWA_CIFAR10')

    # Data
    print('[*] preparing data')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('[*] building model')
    model = ResNet18()
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Optimizer
    epochs = 2
    params = model.parameters()
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    train(epochs, writer)

    # TensorBoard writer 닫기
    writer.close()


    # directory = './ckpt_0'
    # ckpt_list = os.listdir(directory)
    # ckpt_list = [f for f in ckpt_list if os.path.isfile(os.path.join(directory, f)) and  f.startswith(f'model_SWA_{model.__class__.__name__}')]
    # ckpt_list.sort()
    # ckpt_path = os.path.join(directory, ckpt_list[-1])
    # print(ckpt_path)
    # test(ckpt_path=ckpt_path)

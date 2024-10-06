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

    scaler = GradScaler('cuda')  # Mixed Precision 스케일러

    for epoch in range(1, epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_loss = 0.0  # 에포크별로 총 손실을 저장하기 위한 변수
        total_acc = 0.0  # 에포크별로 총 정확도를 저장하기 위한 변수
        for step, (data, targets) in enumerate(trainloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):  # Mixed precision 적용
                outputs = model(data)
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, targets)

            scaler.scale(loss).backward()  # Mixed precision 스케일링된 그라디언트 적용
            scaler.step(optimizer)  # 옵티마이저 스텝
            scaler.update()  # 스케일 업데이트

            # 손실 및 정확도 계산
            running_loss += loss.item()
            total_loss += loss.item()  # 에포크별 총 손실을 계산
            acc, _ = accuracy(outputs, targets)
            running_acc += acc[0].item()
            total_acc += acc[0].item()  # 에포크별 총 정확도 계산

            # 매 100 스텝마다 손실과 정확도 출력 및 기록
            if step % 20 == 0:
                avg_loss = running_loss / (step + 1)
                avg_acc = running_acc / (step + 1)
                print(
                    f'[Epoch {epoch}/{epochs}, Step {step}/{len(trainloader)}] Loss {avg_loss:.4f}, Accuracy {avg_acc:.4f}')

                # TensorBoard에 손실 및 정확도 기록
                writer.add_scalar('Loss/train_20step', avg_loss, epoch * len(trainloader) + step)
                writer.add_scalar('Accuracy/train_20step', avg_acc, epoch * len(trainloader) + step)



        # 에포크 끝난 후 에포크별 평균 손실 및 정확도 계산
        avg_epoch_loss = total_loss / len(trainloader)  # 에포크 동안의 평균 손실
        avg_epoch_acc = total_acc / len(trainloader)  # 에포크 동안의 평균 정확도
        print(f'[Epoch {epoch}/{epochs}] Epoch Loss: {avg_epoch_loss:.4f}, Epoch Accuracy: {avg_epoch_acc:.4f}')

        # TensorBoard에 에포크별 평균 손실 및 정확도 기록
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', avg_epoch_acc, epoch)
        # Scheduler step
        scheduler.step()

        # TensorBoard에 학습률 기록
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        total_cor = 0
        total_samples = 0
        total_test_loss = 0.0  # 테스트 손실을 저장하기 위한 변수

        with torch.no_grad():
            for step, (data, targets) in enumerate(testloader):
                data = data.to(device, dtype=torch.float)
                targets = targets.to(device)
                outputs = model(data)

                # 테스트 손실 계산 및 누적
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, targets)
                total_test_loss += loss.item()  # 테스트 손실 누적

                _, num_cor = accuracy(outputs, targets)
                num_cor = num_cor[0].item()
                total_samples += data.size(0)
                total_cor += num_cor

            # 에포크별 평균 테스트 손실 및 정확도 계산
            avg_test_loss = total_test_loss / len(testloader)  # 테스트 평균 손실
            acc = total_cor / total_samples
            print(f'Epoch {epoch} : Test Accuracy {acc:.4f}, Test Loss {avg_test_loss:.4f}')

            # TensorBoard에 에포크별 Test Accuracy 및 Test Loss 기록
            writer.add_scalar('Accuracy/test_epoch', acc, epoch)
            writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)

        if acc > best_acc:
            print('[*] model saving...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('ckpt_baseline'):
                os.mkdir('ckpt_baseline')

            path = f'ckpt_baseline/model_{model.__class__.__name__}_state_{epoch:03d}_{acc:.4f}.st'
            torch.save(state, path)
            #torch.save(state, path, _use_new_zipfile_serialization=False)
            best_acc = acc
    print(f'Best Test Accuracy {best_acc:.4f}')

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

        # TensorBoard에 test accuracy 기록
        writer.add_scalar('Test Accuracy', acc)

        # Visualize
        images = []
        pred_classes = []
        labels = []
        pred = outputs.topk(1, dim=1, largest=True, sorted=True)
        fig, axes = plt.subplots(3, 3, figsize=(15, 5))  # 3 row, 3 columns
        axes = axes.flatten()
        for k in range(9):  # check only the first 9 images
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

    # GPU 성능 최적화를 위한 설정
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # 최적화된 GPU 성능을 위한 설정

    # TensorBoard writer 설정 (고유한 실험명 사용)
    experiment_name = 'resnet18_baseline_epoch50_20interval'  # 직접 입력한 실험명
    log_dir = f'./runs/{experiment_name}'  # 고유한 로그 디렉토리 이름 생성
    writer = SummaryWriter(log_dir=log_dir)  # 해당 디렉토리에 기록



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
        trainset, batch_size=512, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('[*] building model')
    # model = ToyNetwork()
    model = ResNet18()
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Optimizer
    epochs = 50
    params = model.parameters()
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    train(epochs)

    # Test
    # directory = './ckpt_baseline'
    # ckpt_list = os.listdir(directory)
    # ckpt_list = [f for f in ckpt_list if os.path.isfile(os.path.join(directory, f)) and model.__class__.__name__ in f]
    # ckpt_list.sort()
    # ckpt_path = os.path.join(directory, ckpt_list[-1])
    # print(ckpt_path)
    # test(ckpt_path=ckpt_path)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from models.resnet import *
from torch.utils.tensorboard import SummaryWriter

def rand_bbox(size, lam):
    W = size[2]  #입력 이미지의 너비
    H = size[3]  #높이
    cut_rat = np.sqrt(1. - lam) #lambda 값에 따라 잘라낸 영역 비율
    cut_w = int(W * cut_rat) #비율을 이용하여 잘라낼 너비
    cut_h = int(H * cut_rat) # 높이

    # 이미지의 범위 내에서 잘라낼 영역의 중심
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W) #잘라낼 영역 좌상단x좌표
    bby1 = np.clip(cy - cut_h // 2, 0, H) #좌상단 y좌표
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  #우하단 x좌표
    bby2 = np.clip(cy + cut_h // 2, 0, H)  #우하단 y좌표

    return bbx1, bby1, bbx2, bby2 #좌표 반환

def accuracy(output, target, topk=(1,)):
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
            acc.append(correct_k.mul_(1.0 / batch_size))
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

def train(epochs):
    best_acc = 0.0
    beta = 1.0
    cutmix_prob = 0.5 # cutmix가 적용되는 확률
    test_acc_history = []
    print('[*] 훈련 시작')



    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0  # 에포크 당 손실을 기록할 변수
        total_acc = 0.0  # 에포크 당 정확도를 기록할 변수
        total_samples = 0  # 총 샘플 수

        for step, (data, targets) in enumerate(trainloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            #cutmix 적용
            r = np.random.rand(1) # 0과 1사이의 난수 생성
            if beta > 0.0 and r < cutmix_prob:
                # CutMix 적용
                lam = np.random.beta(beta, beta) #람다값 샘플링
                rand_index = torch.randperm(data.size()[0]).to(device) #데이터 셋의 인덱스를 섞음
                target_a = targets #원본 타겟
                target_b = targets[rand_index] # cutmix에 의해 랜덤하게 섞인 타겟을 저장
                #cutmix에 적용할 바운딩 박스
                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2] #박스 영역 내에서 랜덤하게 섞은 인덱스를 이용하여 교체

                # 박스 크기에 따라 람다 값 다시 계산
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

                # Compute output
                output = model(data)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam) # 손실을 두 타겟(원본이미지,cutmix 적용 이미지)에 대해 가중 합산
            else:
                # 일반 손실 계산
                output = model(data)
                loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 에포크 손실 및 정확도 업데이트
            running_loss += loss.item() * data.size(0)
            acc, _ = accuracy(output, targets)
            total_acc += acc[0].item() * data.size(0)
            total_samples += data.size(0)

            if step % 10 == 0:
                print(
                    f'[Epoch {epoch}/{epochs}, Step {step}/{len(trainloader)}] Loss {loss.item():.4f}, Accuracy {acc[0].item():.4f}')

        scheduler.step()

        # 에포크 손실 및 정확도 기록
        epoch_loss = running_loss / total_samples
        epoch_acc = total_acc / total_samples
        #텐서보드에도 기록
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # 테스트 정확도 기록
        model.eval()
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
            test_acc_history.append(acc)
            writer.add_scalar('Accuracy/test', acc, epoch)  # 테스트 정확도 기록
            print(f'Epoch {epoch} : Test Accuracy {acc:.4f}')

        if acc > best_acc:
            print('[*] 모델 저장 중...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('ckpt_0'):
                os.mkdir('ckpt_0')

            path = f'ckpt_0/model_{model.__class__.__name__}_state_{epoch:03d}_{acc:.4f}.st'
            torch.save(state, path)
            best_acc = acc

    print(f'최고 테스트 정확도 {best_acc:.4f}')

    writer.close()  # writer 종료


def test(ckpt_path):
    print(f'[*] {ckpt_path} 로드 중')
    model.eval()
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'], strict=True)

    total_cor = 0
    total_samples = 0
    mixed_images = []
    pred_classes = []
    labels = []

    with torch.no_grad():
        for step, (data, targets) in enumerate(testloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            outputs = model(data)
            _, num_cor = accuracy(outputs, targets)
            num_cor = num_cor[0].item()
            total_samples += data.size(0)
            total_cor += num_cor

            # CutMix 적용된 데이터 시각화
            if step < 1:  # 첫 번째 배치만 시각화
                for i in range(data.size(0)):
                    # 실제 데이터와 함께 교체된 이미지 저장
                    mixed_images.append(data[i].detach().cpu().permute(1, 2, 0).numpy())
                    pred_classes.append(classes[outputs[i].argmax().item()])
                    labels.append(classes[targets[i].item()])

        acc = total_cor / total_samples
        print(f'로드한 모델 {model.__class__.__name__}의 테스트 정확도 {acc:.4f}')

    # 시각화
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3행 3열
    axes = axes.flatten()
    for k in range(9):  # 처음 9개 이미지만 확인
        axes[k].imshow(mixed_images[k])  # CutMix된 이미지
        axes[k].axis('off')
        axes[k].set_title(f'label: {labels[k]},pred: {pred_classes[k]} ', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    writer = SummaryWriter(log_dir='runs/CutMix_CIFAR10')  # 텐서보드 writer 초기화
    # 데이터 준비
    print('[*] 데이터 준비 중')
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

    # 모델 구축
    print('[*] 모델 구축 중')

    model = ResNet18()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 옵티마이저
    epochs = 50
    params = model.parameters()
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 훈련
   # train(epochs)


    directory = './ckpt_0'
    ckpt_list = os.listdir(directory)
    ckpt_list = [f for f in ckpt_list if os.path.isfile(os.path.join(directory, f)) and model.__class__.__name__ in f]
    ckpt_list.sort()
    ckpt_path = os.path.join(directory, ckpt_list[-1])
    print(ckpt_path)
    test(ckpt_path=ckpt_path)

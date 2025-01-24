---
layout: post
title:  "An example to learn knowledge distillation"
date:   2025-01-24 10:01:36 +0800
categories: deep learning
---
本文通过一个图像分类的例子练习一下如何做知识蒸馏。

数据集选用CIFAR10.
知识蒸馏是用一个比较大的教师模型的output来训练较小的学生模型，paper可以参考[这篇](https://blog.csdn.net/level_code/article/details/135916505)。

本文用一个简单的教师模型resnet18，resnet18本来用于imageNet, 现在cifar10的图片太小，需要修改一下防止图片被下采样没信息了，改卷积层的kernel size和maxpool layer.  另外，不要直接调用model.forward, 那样会调用修改之前的模型，还是会报错，因为修改之前的resnet18针对的是imageNet的图片尺寸，不适用于cifar10.

```python
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        # 修改第一层以适应32x32输入
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        # 修改最后一层以适应CIFAR-10的10个类别
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
```
先把这个教师模型训练出来。加上early stop机制。

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
def train_teacher(train_loader, val_loader, model, epochs=50, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=5)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99:
                print(
                    f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f} acc: {100. * correct / total:.2f}%')
                running_loss = 0.0

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_teacher.pth')

        scheduler.step(val_loss)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

    return train_losses, train_accs, val_losses, val_accs
```
接下来用这个教师模型去指导一个小型的学生模型。
定义一个只有3个卷积层的学生模型。

```python
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

定义知识蒸馏的损失函数，损失包含hard label部分和soft label部分，hard label是标注的label, soft label是教师模型的output经过温度的变化。
而且要注意，由于soft target产生的梯度幅度 相当于缩放了 1/T2 ，因此在同时使用hard 和 soft targets时将其乘以 T^2^.
其中hard loss用的是CE loss, soft loss用的是KL散度。

```python
class DistillationLoss(nn.Module):
    def __init__(self, T=4):
        super(DistillationLoss, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits, labels, alpha=0.5):
        soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=1)
        soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=1)
        #KL散度：Loss = -Σ P(x) * log(Q(x))    [忽略常数项 Σ P(x) * log(P(x))]
        #除以soft_prob.size(0)是为了取平均，使得批大小的改变不会影响损失的规模
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size(0)

        hard_loss = nn.functional.cross_entropy(student_logits, labels)
        #用温度T软化概率分布时，梯度会被同时缩小T^2倍
        #为了保持梯度的数量级与硬目标损失相近，需要乘上T^2进行补偿
        loss = alpha * hard_loss + (1 - alpha) * (self.T ** 2) * soft_targets_loss
        return loss
```
知识蒸馏训练

```python
def main():
    set_seed(225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    teacher_model = TeacherNet().to(device)
    student_model = StudentNet().to(device)

    teacher_weights = torch.load('best_teacher.pth', map_location=device)
    teacher_model.load_state_dict(teacher_weights)
    print("成功加载教师模型权重！")

    train_student(teacher_model, student_model, trainloader, epochs=15, device=device)

    teacher_acc = evaluate(teacher_model, testloader, device)
    student_acc = evaluate(student_model, testloader, device)

    print(f"教师网络测试准确率: {teacher_acc:.2f}%")
    print(f"知识蒸馏后的学生网络测试准确率: {student_acc:.2f}%")

    # 保存学生模型权重
    torch.save(student_model.state_dict(), 'student_distill.pth')

if __name__ == '__main__':
    main()
```
你可以通过改变温度T，hard loss和soft loss的占比alpha来看不同的温度T和不同的比例对知识蒸馏的影响。

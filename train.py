import scipy.io
import numpy as np
import torch
import torch.nn as nn
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Baseline

# 读取.mat文件
mat_file = scipy.io.loadmat('Mult-class Problem.mat')

train_data = mat_file['Training_data']
test_data = mat_file['Testing_data']
train_mask = mat_file['Label_training']
test_mask = mat_file['Label_testing']

#获取总类别数 训练样本数 测试样本数 维度n
sum_classes = max(train_mask[0])
train_size = train_mask[0].size
test_size = test_mask[0].size
n = train_data[:,0].size

#模型保存路径
model_path = "models\\baseline.pth"
#模型加载路径
pre_model_path = "models\\baseline.pth"

#制作数据集
train_dataset = MyDataset(train_data,train_mask)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义模型和优化器
model = Baseline(sum_classes)
#预加载之前的模型
model.load_state_dict(torch.load(pre_model_path, map_location='cpu'))

criterion = nn.CrossEntropyLoss().cuda()
learning_rate = 1e-3
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

model.cuda()

best_acc = 0

epochs = 50

for epoch in range (epochs):
    model.train()
    losses = []
    for i,(input, label) in enumerate(train_dataloader):
        input = input.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        outputs = model(input)
        loss = criterion(outputs, label)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 将损失值保存并打印出本轮的平均损失
        losses.append(loss.item())
    epoch_loss = np.mean(losses)
    print("Epoch {}: Average Loss = {:.4f}".format(epoch + 1, epoch_loss))

    # 在训练集上评估模型，并保存性能最好的模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input, label in train_dataloader:
            input = input.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc = 100 * correct / total
    print('The accuracy of the best model on train set: {:.2f}%'.format(acc))
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), model_path)
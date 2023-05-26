import scipy.io
import torch
from MyDataset import MyDataset
from torch.utils.data import DataLoader
from model import Baseline
from collections import Counter

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

#制作数据集
test_dataset = MyDataset(test_data,test_mask)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# 定义模型
model = Baseline(sum_classes)

# 加载模型权重
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.cuda()

# 在测试集上评估模型性能
model.eval()
correct = 0
total = 0
predicted_labels = []
with torch.no_grad():
    for input, label in test_dataloader:
        input = input.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        outputs = model(input)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted.cpu().numpy().tolist())
        total += label.size(0)
        correct += (predicted == label).sum().item()

counter = Counter(predicted_labels)
for num, count in counter.most_common():
    print(f"Class {num+1} appears {count} times")

acc = 100 * correct / total
print('The accuracy of the best model on test set: {:.2f}%'.format(acc))
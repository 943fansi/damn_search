import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

NUM = 4000

# rand_value=np.random.randint(low=1,high=4000,size=NUM )
# print(rand_value.shape, rand_value[0])
rand_value=list(range(NUM))

# 类别1： x, x+1, x+2 : 0  // 类别2： x, x, x : 1
group_one = np.zeros([NUM , 4], dtype=np.float) 
group_two = np.ones([NUM , 4], dtype=np.float)
for i in range(len(rand_value)):
    group_one[i,:-1] = np.array([rand_value[i], rand_value[i]+1, rand_value[i]+2])
    group_two[i,:-1] = np.array([rand_value[i], rand_value[i], rand_value[i]])

data = np.vstack([group_one, group_two])
# X_train,X_test, y_train, y_test =train_test_split(data[:,:-1],data[:,-1],test_size=0.4, random_state=0)
X_train, y_train = data[:,:-1],data[:,-1]
print(X_train[0], y_train[0])

# 归一化
scale = StandardScaler(with_mean=True,with_std=True)
X_train_norm = scale.fit_transform(X_train)
# X_test_norm = scale.transform(X_test)

xtrain = torch.tensor(X_train_norm, dtype=torch.float32)
ytrain = torch.tensor(y_train).type(torch.LongTensor)
# xtest = torch.tensor(X_test_norm, dtype=torch.float32)
# ytest = torch.tensor(y_test).type(torch.LongTensor)


trainSet = Data.TensorDataset(xtrain, ytrain)
train_loader = Data.DataLoader(
    dataset=trainSet,  ## 使用的数据集
    batch_size=64,  # 批处理样本大小
    shuffle=True,  # 每次迭代前打乱数据
    drop_last=True
)

## 使用继承Module的形式定义全连接神经网络
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(
            in_features=3,  ## 第一个隐藏层的输入，数据的特征数
            out_features=64,  ## 第一个隐藏层的输出，神经元的数量
            bias=True,  ## 默认会有偏置
        )
        self.active1 = nn.ReLU()
        ## 定义第一个隐藏层
        self.hidden2 = nn.Linear(64, 10)
        self.active2 = nn.ReLU()
        ## 定义预测回归层
        self.logits = nn.Linear(10, 2)

    ## 定义网络的向前传播路径
    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        x = self.logits(x)
        output =torch.squeeze(x)
        ## 输出为output
        return output

model = MLPmodel().to(device)
print(model)

loss_func = nn.CrossEntropyLoss()  # 交叉熵
optimizer = SGD(model.parameters(), lr=0.001)

model.train()
train_loss_all = []  ## 输出每个批次训练的损失函数 .to(torch.float32)
## 进行训练，并输出每次迭代的损失函数
for epoch in range(30):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        output = model(b_x)  # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)  #
 
        optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
        train_loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        train_loss_all.append(train_loss.item())

plt.figure()
plt.plot(train_loss_all, "r-")
plt.title("Train loss per iteration")
plt.show()


# test
single_x = np.array([1, 1, 1]).reshape(-1, 3)
single_x_norm = scale.transform(single_x)
single_testx=torch.tensor(single_x_norm, dtype=torch.float32).view(-1, 1, 3).to(device)

# _x = np.array([10, 11, 12]).reshape(-1, 3)
# _x_norm = scale.transform(_x)
# _testx=torch.tensor(_x_norm, dtype=torch.float32).view(-1, 1, 3).to(device)

model.eval()
with torch.no_grad():
    out_1 = model(single_testx).cpu()
    predicted_1 =out_1.argmax(0)

    # out_2 = model(_testx).cpu()
    # predicted_2 =out_2.argmax(0)

print(predicted_1)
# print(predicted_1, predicted_2)


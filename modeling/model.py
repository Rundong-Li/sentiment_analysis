import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.config import *


class Model(nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.n_class = 2

    def forward(self, x):
        pass


class TextCNN(Model):
    def __init__(self, vocab_size, word2vec):
        super(TextCNN, self).__init__(vocab_size)
        self.embedding_dim = 50
        self.drop_keep_prob = 0.5
        self.pretrained_embed = word2vec
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        # 使用预训练的词向量
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_embed))
        self.embedding.weight.requires_grad = True
        # 卷积层
        self.conv1 = nn.Conv2d(1, kernel_num, (kernel_size[0], self.embedding_dim))
        self.conv2 = nn.Conv2d(1, kernel_num, (kernel_size[1], self.embedding_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (kernel_size[2], self.embedding_dim))
        # Dropout
        self.dropout = nn.Dropout(self.drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(len(self.kernel_size) * self.kernel_num, self.n_class)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length, embedding_dim)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv2)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv3)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, dataloader, epoch, optimizer, criterion, scheduler):
    # 定义训练过程
    train_loss, train_acc = 0.0, 0.0
    count, correct = 0, 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)
        """
        if (batch_idx + 1) % 10 == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
        """

    train_loss *= BATCH_SIZE
    train_loss /= len(dataloader.dataset)
    train_acc = correct / count
    print('\ntrain epoch: {}\taverage loss: {:.6f}\taccuracy:{:.4f}%\n'.format(epoch, train_loss, 100. * train_acc))
    scheduler.step()

    return train_loss, train_acc


def validation(model, dataloader, epoch, criterion):
    model.eval()
    # 验证过程
    val_loss, val_acc = 0.0, 0.0
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

    val_loss *= BATCH_SIZE
    val_loss /= len(dataloader.dataset)
    val_acc = correct / count
    # 打印准确率
    print(
        'validation:train epoch: {}\taverage loss: {:.6f}\t accuracy:{:.2f}%\n'.format(epoch, val_loss, 100 * val_acc))

    return val_loss, val_acc


def test(model, dataloader):
    model.eval()
    model.to(DEVICE)

    # 测试过程
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

    # 打印准确率
    print('test accuracy:{:.2f}%.'.format(100 * correct / count))

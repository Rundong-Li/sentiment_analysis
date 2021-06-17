import torch
learning_rate = 0.01  # 学习率
BATCH_SIZE = 512  # 训练批量
EPOCHS = 2  # 训练轮数
# model_path = "./saved_models/"  # 预训练模型路径
DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
# update_w2v = True  # 是否在训练中更新w2v
n_class = 2  # 分类数：分别为pos和neg
vocab_size = 70000  # 词汇表大小
padding_idx = 0  # '_PAD_' token 对应的idx
hid_dim = 512  # embedding层隐向量维度
n_layers = 12   # EncoderLayer层数
n_heads = 8    # 多头自注意力head数
pf_dim = 2048   # PositionwiseFeedFoward中间层的维度
dropout = 0.3  # dropout层
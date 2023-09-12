import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import numpy as np
from model import *

# 定义数据集转换函数，用于将文本数据转换为张量数据，并生成负样本
def convert_data(data):
    head, relation, tail = data["text"].asnumpy().decode().split()
    head_id = np.array([entity2id[head]], dtype=np.int32)
    relation_id = np.array([relation2id[relation]], dtype=np.int32)
    tail_id = np.array([entity2id[tail]], dtype=np.int32)
    # 随机选择一个实体作为负样本，替换头实体或尾实体
    negative_id = np.random.randint(len(entity2id), size=1, dtype=np.int32)
    mode = np.random.randint(2, size=1, dtype=np.int32) # 0表示替换头实体，1表示替换尾实体
    if mode == 0:
        return (head_id, relation_id, tail_id, negative_id, tail_id, mode)
    else:
        return (head_id, relation_id, tail_id, head_id, negative_id, mode)

# 定义训练函数，输入为模型、优化器、损失函数和数据集
def train(model, optimizer, loss_fn, dataset):
    # 遍历训练轮数
    for epoch in range(num_epochs):
        # 遍历数据集中的每个批次
        for data in dataset.create_dict_iterator():
            # 获取正样本和负样本的头实体、关系和尾实体的编号
            head_id = data["head"].asnumpy()
            relation_id = data["relation"].asnumpy()
            tail_id = data["tail"].asnumpy()
            negative_head_id = data["negative_head"].asnumpy()
            negative_tail_id = data["negative_tail"].asnumpy()
            mode = data["mode"].asnumpy()
            # 计算模型的输出，即正负样本的得分或损失
            output = model(head_id, relation_id, tail_id, negative_head_id, negative_tail_id, mode)
            # 计算损失值
            if loss_fn == nn.ReduceMean(): # 对于TransE和RotatE，直接使用输出作为损失值
                loss = loss_fn(output)
            else: # 对于DistMult和RESCAL，使用Softmax交叉熵作为损失函数
                label = np.concatenate([tail_id, head_id], axis=0) # 拼接正样本的尾实体和头实体作为标签
                loss = loss_fn(output, label)
            # 反向传播更新参数
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
        # 在每个训练轮结束后，打印当前的损失值
        print("Epoch:", epoch, "Loss:", loss.asnumpy())

# 定义MRR，hit@1，hit@10指标的计算函数
def evaluate(model, dataset):
    # 初始化指标的累积值和计数值
    mrr = 0.0
    hit_1 = 0
    hit_10 = 0
    count = 0
    # 遍历数据集中的每个三元组
    for data in dataset.create_dict_iterator():
        # 获取头实体、关系和尾实体的编号
        head_id = data["head"].asnumpy()
        relation_id = data["relation"].asnumpy()
        tail_id = data["tail"].asnumpy()
        # 计算给定头实体和关系时，所有实体作为尾实体的得分
        tail_score = model(head_id, relation_id, np.arange(len(entity2id), dtype=np.int32))
        # 计算给定尾实体和关系时，所有实体作为头实体的得分
        head_score = model(np.arange(len(entity2id), dtype=np.int32), relation_id, tail_id)
        # 对两个得分进行拼接，得到一个二维数组，每一行表示一个替换实体的情况
        score = np.concatenate([tail_score, head_score], axis=0)
        # 对得分进行降序排序，并获取排序后的索引
        rank = np.argsort(-score, axis=1)
        # 计算原始三元组在排序中的位置，即正确实体的排名
        tail_rank = np.where(rank == tail_id)[1] + 1 # 加1是为了从1开始计数
        head_rank = np.where(rank == head_id)[1] + 1
        # 更新指标的累积值和计数值
        mrr += (1 / tail_rank + 1 / head_rank).mean() # 平均每个替换实体的情况下的MRR
        hit_1 += ((tail_rank <= 1) | (head_rank <= 1)).mean() # 平均每个替换实体的情况下的hit@1
        hit_10 += ((tail_rank <= 10) | (head_rank <= 10)).mean() # 平均每个替换实体的情况下的hit@10
        count += 1
    # 计算指标的平均值并返回
    return mrr / count, hit_1 / count, hit_10 / count


# 定义超参数
batch_size = 128 # 批次大小
embedding_dim = 128 # 实体和关系的嵌入维度
learning_rate = 0.01 # 学习率
num_epochs = 100 # 训练轮数
margin = 1.0 # TransE的边界值
gamma = 12.0 # RotatE的尺度因子

# 加载family数据集，该数据集包含14个实体和3个关系，共有104条三元组
# 数据集的格式为(head, relation, tail)
family_dataset = ds.TextFileDataset("family.txt", shuffle=True)

# 构建实体和关系的索引字典，用于将名称转换为整数编号
entity2id = {}
relation2id = {}
for data in family_dataset.create_dict_iterator():
    head, relation, tail = data["text"].asnumpy().decode().split()
    if head not in entity2id:
        entity2id[head] = len(entity2id)
    if relation not in relation2id:
        relation2id[relation] = len(relation2id)
    if tail not in entity2id:
        entity2id[tail] = len(entity2id)


# 对数据集进行转换，并设置批次大小
family_dataset = family_dataset.map(operations=convert_data,
                                    input_columns=["text"],
                                    output_columns=["head", "relation", "tail", "negative_head", "negative_tail", "mode"],
                                    column_order=["head", "relation", "tail", "negative_head", "negative_tail", "mode"])
family_dataset = family_dataset.batch(batch_size)


# 实例化TransE模型，并设置优化器和损失函数
transe_model = TransE(len(entity2id), len(relation2id), embedding_dim, margin)
transe_optimizer = nn.Adam(transe_model.trainable_params(), learning_rate=learning_rate)
transe_loss_fn = nn.ReduceMean()

# 实例化DistMult模型，并设置优化器和损失函数
distmult_model = DistMult(len(entity2id), len(relation2id), embedding_dim)
distmult_optimizer = nn.Adam(distmult_model.trainable_params(), learning_rate=learning_rate)
distmult_loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

# 实例化RESCAL模型，并设置优化器和损失函数
rescal_model = RESCAL(len(entity2id), len(relation2id), embedding_dim)
rescal_optimizer = nn.Adam(rescal_model.trainable_params(), learning_rate=learning_rate)
rescal_loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

# 实例化RotatE模型，并设置优化器和损失函数
rotate_model = RotatE(len(entity2id), len(relation2id), embedding_dim, gamma)
rotate_optimizer = nn.Adam(rotate_model.trainable_params(), learning_rate=learning_rate)
rotate_loss_fn = nn.ReduceMean()



# 训练TransE模型，并评估其性能
print("Training TransE model...")
train(transe_model, transe_optimizer, transe_loss_fn, family_dataset)
print("Evaluating TransE model...")
mrr, hit_1, hit_10 = evaluate(transe_model, family_dataset)
print("MRR:", mrr, "Hit@1:", hit_1, "Hit@10:", hit_10)

# 训练DistMult模型，并评估其性能
print("Training DistMult model...")
train(distmult_model, distmult_optimizer, distmult_loss_fn, family_dataset)
print("Evaluating DistMult model...")
mrr, hit_1, hit_10 = evaluate(distmult_model, family_dataset)
print("MRR:", mrr, "Hit@1:", hit_1, "Hit@10:", hit_10)

# 训练RESCAL模型，并评估其性能
print("Training RESCAL model...")
train(rescal_model, rescal_optimizer, rescal_loss_fn, family_dataset)
print("Evaluating RESCAL model...")
mrr, hit_1, hit_10 = evaluate(rescal_model, family_dataset)
print("MRR:", mrr, "Hit@1:", hit_1, "Hit@10:", hit_10)

# 训练RotatE模型，并评估其性能
print("Training RotatE model...")
train(rotate_model, rotate_optimizer, rotate_loss_fn, family_dataset)
print("Evaluating RotatE model...")
mrr, hit_1, hit_10 = evaluate(rotate_model, family_dataset)
print("MRR:", mrr, "Hit@1:", hit_1, "Hit@10:", hit_10)

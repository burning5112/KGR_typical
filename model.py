import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import numpy as np


# 定义TransE模型类，继承自nn.Cell
class TransE(nn.Cell):
    def __init__(self, num_entity, num_relation, embedding_dim, margin):
        super(TransE, self).__init__()
        self.num_entity = num_entity  # 实体数量
        self.num_relation = num_relation  # 关系数量
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.margin = margin  # 边界值
        # 定义实体嵌入矩阵，初始化为[-6/根号(d), 6/根号(d)]之间的均匀分布，其中d是嵌入维度
        self.entity_embedding = nn.Embedding(num_entity, embedding_dim,
                                             embedding_table=ops.init.initializer(
                                                 'uniform', [num_entity, embedding_dim], -6 / np.sqrt(embedding_dim),
                                                                                         6 / np.sqrt(embedding_dim)))
        # 定义关系嵌入矩阵，初始化为[-6/根号(d), 6/根号(d)]之间的均匀分布，其中d是嵌入维度
        self.relation_embedding = nn.Embedding(num_relation, embedding_dim,
                                               embedding_table=ops.init.initializer(
                                                   'uniform', [num_relation, embedding_dim],
                                                   -6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim)))
        # 定义L2范数操作
        self.l2_norm = ops.L2Normalize(axis=-1)
        # 定义平方和操作
        self.squared_sum = ops.ReduceSum(keep_dims=False)
        # 定义最大值操作
        self.maximum = ops.Maximum()

    def construct(self, head_id, relation_id, tail_id, negative_head_id, negative_tail_id, mode):
        # 查找正样本的头实体、关系和尾实体的嵌入向量，并进行L2范数归一化
        head = self.l2_norm(self.entity_embedding(head_id))
        relation = self.l2_norm(self.relation_embedding(relation_id))
        tail = self.l2_norm(self.entity_embedding(tail_id))
        # 查找负样本的头实体和尾实体的嵌入向量，并进行L2范数归一化
        negative_head = self.l2_norm(self.entity_embedding(negative_head_id))
        negative_tail = self.l2_norm(self.entity_embedding(negative_tail_id))
        # 根据负采样的模式，计算正负样本的得分
        if mode == 0:  # 替换头实体
            positive_score = head + relation - tail  # 正样本的得分为h+r-t
            negative_score = negative_head + relation - tail  # 负样本的得分为h'+r-t
        else:  # 替换尾实体
            positive_score = tail - head - relation  # 正样本的得分为t-h-r
            negative_score = negative_tail - head - relation  # 负样本的得分为t'-h-r
        # 计算正负样本的得分的L2范数的平方
        positive_score = self.squared_sum(positive_score ** 2, -1)
        negative_score = self.squared_sum(negative_score ** 2, -1)
        # 计算正负样本之间的边界损失
        loss = self.maximum(positive_score - negative_score + self.margin, 0.0)
        return loss


# 定义DistMult模型类，继承自nn.Cell
class DistMult(nn.Cell):
    def __init__(self, num_entity, num_relation, embedding_dim):
        super(DistMult, self).__init__()
        self.num_entity = num_entity  # 实体数量
        self.num_relation = num_relation  # 关系数量
        self.embedding_dim = embedding_dim  # 嵌入维度
        # 定义实体嵌入矩阵，初始化为[-1/根号(d), 1/根号(d)]之间的均匀分布，其中d是嵌入维度
        self.entity_embedding = nn.Embedding(num_entity, embedding_dim,
                                             embedding_table=ops.init.initializer(
                                                 'uniform', [num_entity, embedding_dim], -1 / np.sqrt(embedding_dim),
                                                                                         1 / np.sqrt(embedding_dim)))
        # 定义关系嵌入矩阵，初始化为[-1/根号(d), 1/根号(d)]之间的均匀分布，其中d是嵌入维度
        self.relation_embedding = nn.Embedding(num_relation, embedding_dim,
                                               embedding_table=ops.init.initializer(
                                                   'uniform', [num_relation, embedding_dim],
                                                   -1 / np.sqrt(embedding_dim), 1 / np.sqrt(embedding_dim)))
        # 定义元素积操作
        self.multiply = ops.Mul()
        # 定义求和操作
        self.sum = ops.ReduceSum(keep_dims=False)

    def construct(self, head_id, relation_id, tail_id):
        # 查找头实体、关系和尾实体的嵌入向量
        head = self.entity_embedding(head_id)
        relation = self.relation_embedding(relation_id)
        tail = self.entity_embedding(tail_id)
        # 计算三元组的得分为h*r*t，即头实体、关系和尾实体嵌入向量的元素积之和
        score = self.sum(self.multiply(head, self.multiply(relation, tail)), -1)
        return score


# 定义RESCAL模型类，继承自nn.Cell
class RESCAL(nn.Cell):
    def __init__(self, num_entity, num_relation, embedding_dim):
        super(RESCAL, self).__init__()
        self.num_entity = num_entity  # 实体数量
        self.num_relation = num_relation  # 关系数量
        self.embedding_dim = embedding_dim  # 嵌入维度
        # 定义实体嵌入矩阵，初始化为[-1/根号(d), 1/根号(d)]之间的均匀分布，其中d是嵌入维度
        self.entity_embedding = nn.Embedding(num_entity, embedding_dim,
                                             embedding_table=ops.init.initializer(
                                                 'uniform', [num_entity, embedding_dim], -1 / np.sqrt(embedding_dim),
                                                                                         1 / np.sqrt(embedding_dim)))
        # 定义关系嵌入矩阵，初始化为[-1/根号(d), 1/根号(d)]之间的均匀分布，其中d是嵌入维度的平方
        self.relation_embedding = nn.Embedding(num_relation, embedding_dim ** 2,
                                               embedding_table=ops.init.initializer(
                                                   'uniform', [num_relation, embedding_dim ** 2],
                                                   -1 / np.sqrt(embedding_dim ** 2), 1 / np.sqrt(embedding_dim ** 2)))
        # 定义矩阵乘法操作
        self.matmul = ops.MatMul()
        # 定义转置操作
        self.transpose = ops.Transpose()
        # 定义求和操作
        self.sum = ops.ReduceSum(keep_dims=False)

    def construct(self, head_id, relation_id, tail_id):
        # 查找头实体、关系和尾实体的嵌入向量
        head = self.entity_embedding(head_id)
        relation = self.relation_embedding(relation_id)
        tail = self.entity_embedding(tail_id)
        # 将关系嵌入向量调整为方阵的形状
        relation = ops.reshape(relation, (-1, self.embedding_dim, self.embedding_dim))
        # 计算三元组的得分为h*M*r，即头实体嵌入向量与关系矩阵的乘积与尾实体嵌入向量的转置的乘积之和
        score = self.sum(self.matmul(self.matmul(head, relation), self.transpose(tail)), -1)
        return score


# 定义RotatE模型类，继承自nn.Cell
class RotatE(nn.Cell):
    def __init__(self, num_entity, num_relation, embedding_dim, gamma):
        super(RotatE, self).__init__()
        self.num_entity = num_entity  # 实体数量
        self.num_relation = num_relation  # 关系数量
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.gamma = gamma  # 尺度因子
        # 定义实体嵌入矩阵，初始化为[-1/根号(d), 1/根号(d)]之间的均匀分布，其中d是嵌入维度
        self.entity_embedding = nn.Embedding(num_entity, embedding_dim,
                                             embedding_table=ops.init.initializer(
                                                 'uniform', [num_entity, embedding_dim], -1 / np.sqrt(embedding_dim),
                                                                                         1 / np.sqrt(embedding_dim)))
        # 定义关系嵌入矩阵，初始化为[-pi, pi]之间的均匀分布
        self.relation_embedding = nn.Embedding(num_relation, embedding_dim,
                                               embedding_table=ops.init.initializer(
                                                   'uniform', [num_relation, embedding_dim], -np.pi, np.pi))
        # 定义复数乘法操作
        self.complex_mul = ops.ComplexMul()
        # 定义复数模操作
        self.complex_abs = ops.ComplexAbs()
        # 定义L2范数操作
        self.l2_norm = ops.L2Normalize(axis=-1)

    def construct(self, head_id, relation_id, tail_id):
        # 查找头实体、关系和尾实体的嵌入向量，并进行L2范数归一化
        head = self.l2_norm(self.entity_embedding(head_id))
        relation = self.relation_embedding(relation_id)
        tail = self.l2_norm(self.entity_embedding(tail_id))
        # 将头实体和尾实体的嵌入向量分解为实部和虚部，构成复数形式
        head_real = head[..., :self.embedding_dim // 2]
        head_imag = head[..., self.embedding_dim // 2:]
        tail_real = tail[..., :self.embedding_dim // 2]
        tail_imag = tail[..., self.embedding_dim // 2:]
        head_complex = (head_real, head_imag)
        tail_complex = (tail_real, tail_imag)
        # 将关系嵌入向量转换为旋转角度，即以复数形式表示的单位向量
        relation_phase = relation / (self.embedding_dim // 2)  # 归一化到[-1, 1]之间
        relation_cos = ops.cos(relation_phase)  # 计算余弦值作为实部
        relation_sin = ops.sin(relation_phase)  # 计算正弦值作为虚部
        relation_complex = (relation_cos, relation_sin)
        # 计算头实体与关系的复数乘积，作为旋转后的头实体
        rotated_head_complex = self.complex_mul(head_complex, relation_complex)
        # 计算旋转后的头实体与尾实体的复数差的模，作为三元组的负得分
        score = -self.gamma + self.complex_abs(self.complex_mul(rotated_head_complex, ops.conj(tail_complex)))
        return score

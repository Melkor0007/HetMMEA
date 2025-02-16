
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from .layers import ProjectionHead
from .EVA_tools import GAT, GCN


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        # assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        # joint_emb = torch.cat(embs, dim=1)

        hidden_states = torch.stack(embs, dim=1)
        joint_emb = hidden_states.mean(dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb
    
    # def forward(self, embs):
    #     # 对权重进行 softmax 标准化
    #     weight_norm = F.softmax(self.weight, dim=0)
        
    #     pooled_embs = []
    #     for idx in range(self.modal_num):
    #         if embs[idx] is not None:
    #             # 对每个模态的嵌入进行 L2 归一化（假设归一化在最后一维进行）
    #             emb = F.normalize(embs[idx], p=2, dim=-1)
    #             # 对序列维度进行平均池化，得到固定长度的向量
    #             pooled = emb.mean(dim=1)  # 结果形状：(batch_size, emb_dim)
    #             # 将池化后的向量乘以对应模态的权重
    #             pooled_embs.append(weight_norm[idx] * pooled)
        
    #     # 将各个模态的向量堆叠成一个张量，形状：(batch_size, modal_num, emb_dim)
    #     hidden_states = torch.stack(pooled_embs, dim=1)
    #     # 在模态维度上取平均，融合成最终的联合嵌入，形状：(batch_size, emb_dim)
    #     joint_emb = hidden_states.mean(dim=1)
    #     return joint_emb


class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units
    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 desc_feature_dim=None,   # 新增描述特征输入维度
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        # Modal Encoder
        #########################
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)
        # 新增描述特征的全连接层，输出维度同char_dim（可根据需要调整）
        self.desc_fc = nn.Linear(desc_feature_dim, char_dim) if desc_feature_dim is not None else None

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        #########################
        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(self.n_units[2], self.n_units[2], self.n_units[2], dropout)

        #########################
        # Fusion Encoder
        #########################
        # 注意：这里的 inner_view_num 需设置为7（对应 img, att, rel, gph, name, char, desc）
        self.fusion = MultiModalFusion(modal_num=self.args.inner_view_num,
                                       with_weight=self.args.with_weight)

    def _emb_generate(self, input_idx, adj, img_features, rel_features, att_features, 
                      name_features=None, char_features=None, desc_features=None,   # 新增desc_features参数
                      entity_noise=None, entity_noise_mask=None):
        if self.args.w_gcn:
            if entity_noise is None and entity_noise_mask is None:
                gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
            else:
                entity_emb = self.entity_emb(input_idx)
                entity_emb[entity_noise_mask] = (1.0 - self.args.mask_ratio * 0.5) * entity_emb[entity_noise_mask] + self.args.mask_ratio * 0.5 * entity_noise[entity_noise_mask]
                gph_emb = self.cross_graph_model(entity_emb, adj)
        else:
            gph_emb = None

        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None
        # 处理描述特征
        if hasattr(self.args, "w_desc") and self.args.w_desc and (desc_features is not None) and (self.desc_fc is not None):
            desc_emb = self.desc_fc(desc_features)
        else:
            desc_emb = None

        return (gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, desc_emb)

    def forward(self,
                gph_emb=None,
                img_emb=None,
                rel_emb=None,
                att_emb=None,
                name_emb=None,
                char_emb=None,
                desc_emb=None):   # 新增desc_emb参数

        if self.use_project_head:
            gph_emb = self.gph_pro(gph_emb)
            img_emb = self.img_pro(img_emb)
            rel_emb = self.rel_pro(rel_emb)
            att_emb = self.att_pro(att_emb)
            # 注意：通常只对结构、图像、关系和属性进行投影，其它模态（如文本类）可以保持原样

        # 调用融合模块时将所有7个模态传入（顺序需与 MultiModalFusion 保持一致）
        joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb, desc_emb])

        return joint_emb


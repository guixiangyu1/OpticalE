#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

pi = 3.14159262358979323846

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        # gamma 的default是12.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )


        # 初始化embedding
        self.embedding_range = nn.Parameter(
                    torch.Tensor([(5*self.gamma.item() + self.epsilon) / hidden_dim]),
                    requires_grad=False
                )
        # self.embedding_range = nn.Parameter(torch.Tensor([1.0]))
        
        self.entity_dim = (hidden_dim*2) if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'OpticalE', 'rOpticalE', 'TransE_periodic',\
                              'TransE_sin', 'TransE_periodic_2D', 'TransE_periodic_amp', 'TransE_periodic_freq','TransE_periodic_dream', 'TransH_periodic', 'TransH',\
                              'rTransE_periodic_2D', 'resonante', 'TransE_periodic_divide', 'periodic2']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            # unsqueeze(1)在第一个维度处插入维度1: [1,2,3] -> [[1],[2],[3] 3 变成 3*1
            # head.shape batch_size * 1 * embedding_size_for_entity
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            # relation.shape batch_size * 1 * relation_size_for_entity

            # view相当于reshape
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            # tail_shape:  batch_size * negtive_sample_size * entity_embedding_size
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'OpticalE': self.OpticalE,
            'rOpticalE': self.rOpticalE,
            'TransE_periodic': self.TransE_periodic,
            'TransE_sin': self.TransE_sin,
            'TransE_periodic_2D': self.TransE_periodic_2D,
            'TransE_periodic_amp': self.TransE_periodic_amp,
            'TransE_periodic_freq': self.TransE_periodic_freq,
            'TransE_periodic_dream':self.TransE_periodic_dream,
            'TransH_periodic': self.TransH_periodic,
            'TransH': self.TransH,
            'rTransE_periodic_2D': self.rTransE_periodic_2D,
            'resonante': self.resonante,
            'TransE_periodic_divide': self.TransE_periodic_divide,
            'periodic2': self.periodic2
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        # transE 用的是一种概率的log likelihood loss模式，而非原文的那种pairwise的距离loss模式
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score
    # head [16,1,40]; relation [16,1,20]; tail [16,2,40]
    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        #chunk函数是切块用，chunk（tensor，n份，切块的维度），返回tensor的list
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        # re_haed, im_head [16,1,20]; re_tail, im_head [16,2,20]
        #Make phases of relations uniformly distributed in [-pi, pi]

        # phase_relation 属于 正负pi
        phase_relation = relation/(self.embedding_range.item()/pi)
        # re_relation, im_relation [16, 1, 20]
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            # re_score im_score [16,1,20]; re_tail im_tail [16,2,20]
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        # re_score 会 broadcast 成 [16,2,20]
        score = torch.stack([re_score, im_score], dim = 0)
        # score [2,16,2,20]
        #tensor.norm() 求范数；默认是2; 得到的结果往往会删除dim=k的那一维
        score = score.norm(dim = 0)
        # score [16,2,20]

        # 注意，作者将embedding的每一个维度的距离求和，这个和是1范式的，而上面的距离又是二范式的norm
        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def OpticalE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        # re_relation, im_relation [16, 1, 20]
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch': # 逆旋转处理
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score + re_head
            im_score = im_score + im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score + re_tail
            im_score = im_score + im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = score.sum(dim=2) - self.gamma.item()
        return score

    def rOpticalE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        # re_relation, im_relation [16, 1, 20]
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score + re_head
            im_score = im_score + im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score + re_tail
            im_score = im_score + im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def TransE_periodic(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        a = 2 * pi / 3.0
        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        score = (phase_head + phase_relation - phase_tail)

        score = self.square_wave(score)
        # score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2) * self.modulus

        return score

    def TransE_periodic_amp(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        amp_head, phase_head = torch.chunk(head, 2, dim=2)
        amp_tail, phase_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)

        phase = (phase_head + phase_relation - phase_tail)


        score = amp_head**2 + amp_tail**2 + 2 * torch.abs(amp_head * amp_tail) * self.bimodal_sin(phase)
        # score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def TransE_periodic_freq(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        w_r, phi_r = torch.chunk(relation, 2, dim=2)



        # phase_head = phase_head / (self.embedding_range.item() / pi)
        # phase_relation = phi_r / (self.embedding_range.item() / pi)
        # phase_tail = phase_tail / (self.embedding_range.item() / pi)

        phase = w_r * (head + - tail) + phi_r


        score = torch.sin(phase)
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    def TransE_periodic_dream(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        amp_head, phase_head = torch.chunk(head, 2, dim=2)
        amp_tail, phase_tail = torch.chunk(tail, 2, dim=2)
        relation, weight = relation[:,:,:1], relation[:,:,1:]

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)

        w = torch.arange(-0.5 * self.hidden_dim, 0.5 * self.hidden_dim, 1).cuda()
        phase = phase_head + w * phase_relation - phase_tail

        score = (amp_head**2 + amp_tail**2 + 2 * torch.abs(amp_head * amp_tail) * torch.cos(phase)) * weight.abs()


        score = self.gamma.item() - score.sum(dim=2)
        return score

    def TransH_periodic(self, head, relation, tail, mode):
        r, n = torch.chunk(relation, 2, dim=2)
        # h = self._transfer(head, n)
        # t = self._transfer(tail, n)
        # r = self._transfer(r, n)
        phase = head + r - tail
        phase = self._transfer(phase, n)
        score = torch.sin(phase)
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    def TransH(self, head, relation, tail, mode):
        r, n = torch.chunk(relation, 2, dim=2)
        h = self._transfer(head, n)
        t = self._transfer(tail, n)
        score = h+r-t
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


    def _transfer(self, e, n):
        n = F.normalize(n, 2, dim=-1)
        return e - (e * n).sum(dim=2, keepdims=True) * n


    def TransE_periodic_2D(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        x_head, y_head = torch.chunk(head, 2, dim=2)
        x_relation, y_relation = torch.chunk(relation, 2, dim=2)
        x_tail, y_tail = torch.chunk(tail, 2, dim=2)

        x = x_head + x_relation - x_tail
        y = y_head + y_relation - y_tail

        x = x / (self.embedding_range.item() / pi)
        y = y / (self.embedding_range.item() / pi)
        score = torch.sin(x) * torch.sin(y)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2)

        return score

    def rTransE_periodic_2D(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        x_head, y_head = torch.chunk(head, 2, dim=2)
        x_relation, y_relation = torch.chunk(relation, 2, dim=2)
        x_tail, y_tail = torch.chunk(tail, 2, dim=2)

        x = x_head + x_relation - x_tail
        y = y_head + y_relation - y_tail

        x = x / (self.embedding_range.item() / pi)
        y = y / (self.embedding_range.item() / pi)
        score = torch.sin(x) * torch.sin(y)
        score = 1 - torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus

        return score

    def resonante(self, head, relation, tail, mode, train):
        pi = 3.14159262358979323846
        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        phase1 = phase_head + phase_relation - phase_tail
        phase2 = phase_head + 2 * phase_relation - phase_tail

        score1 = torch.abs(torch.sin(phase1))
        score2 = torch.abs(torch.sin(phase2))

        score1 = score1.sum(dim=2)
        score2 = score2.sum(dim=2)

        if train == True:
            score = torch.min(score1, score2 * 2)
        else:
            assert train == False
            score = score1
        score = self.gamma.item() - score * self.modulus

        return score

    def TransE_periodic_divide(self, head, relation, tail, mode):
        # 实体作为头尾分开训练
        pi = 3.14159262358979323846

        head, _ = torch.chunk(head, 2, dim=2)
        _, tail = torch.chunk(tail, 2, dim=2)

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        score = (phase_head + phase_relation - phase_tail)

        score = self.sin(score)
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2) * self.modulus

        return score

    def triangle_cos(self, X):
        pi = 3.14159262358979323846
        return torch.abs(2 / pi * (X % (2*pi) - pi)) - 1


    def triangle_sin(self, X):
        return self.triangle_cos(X - 0.5 * pi)

    def triangle(self, X):
        return  torch.abs(self.triangle_sin(2*X))

    def bimodal(self, a, X):
        T = 2 * pi
        _X = X % T
        mask1 = _X < (0.5 * a)
        mask2 = (_X >= (0.5 * a)) & (_X < a)
        mask3 = (_X >= a) & (_X < (0.5 * (a + T)))
        mask4 = _X >= (0.5 * (a + T))
        X[mask1] = _X[mask1] * (2 / (a))
        X[mask2] = (_X[mask2] - a) * (-2/(a))
        X[mask3] = (_X[mask3] - a) * (2 / (T - a))
        X[mask4] = (_X[mask4] - T) * (-2 / (T -a))
        return X
    def bimodal_sin(self, X):
        T = 2 * pi
        X = X % T
        X = torch.where(X < (2 * pi / 3), torch.sin(3*X), torch.sin(1.5 * (X - 2 * pi / 3)))
        return X

    def quadratic(self, X):
        T = pi
        _X = X % T
        X = (_X - 0.5 * pi) ** 2 * (-4 / pi ** 2) + 1
        return X

    def fourier(self, n, X):
        f = 0.0
        for i in range(n):
            a = 2 * i + 1
            f += torch.sin(a * X) / a
        return abs(4 / pi * f + 1.0)

    def trapezoid_slide(self, X):
        T = 2 * pi
        _X = X % T
        mask1 = _X < (0.5 * pi)
        mask2 = (_X >= (0.5 * pi)) & (_X < 1.5 * pi)
        mask3 = _X >= (1.5 * pi)
        X[mask1] = _X[mask1] * 2 / pi
        X[mask2] = _X[mask2] * 0.01 / pi + 0.995
        X[mask3] = _X[mask3] * (-2.02) / pi + 4.0
        return X

    def trapezoid(self, X):
        T = 2 * pi
        _X = X % T
        mask1 = _X < (pi - 1)
        mask2 = (_X >= (pi - 1)) & (_X < pi + 1)
        mask3 = _X >= (pi + 1)
        X[mask1] = _X[mask1] / (pi - 1)
        X[mask2] = 1.0
        X[mask3] = -(_X[mask3] - 2 * pi) / (pi - 1)
        return X

    def trapezoid_sin(self, X):
        T = 2 * pi
        _X = X % T
        mask1 = _X < 1
        mask2 = (_X >= 1) & (_X < pi - 1)
        mask3 = (_X >= (pi - 1)) & (_X < pi + 1)
        mask4 = (_X >= pi + 1)
        X[mask1] = _X[mask1]
        X[mask2] = 1.0
        X[mask3] = -(_X[mask3] - pi)
        X[mask4] = (_X[mask4] - 2 * pi) / (pi - 1)
        return X

    def square_wave(self, X):
        T = 2 * pi
        _X = X % T
        mask1 = _X < (0.5*pi)
        mask2 = (_X >= (pi*0.5)) & (_X < pi)
        mask3 = (_X >= pi) & (_X < 1.5 * pi)
        mask4 = _X >= 1.5 * pi
        X[mask1] = _X[mask1] * 2 / pi
        X[mask2] = 1.0
        X[mask3] = -(_X[mask3] - 1.5 * pi) * 2 / pi
        X[mask4] = 0.0
        return X

    def TransE_sin(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score) + 1
        # score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def RotatE_topo(self, head, relation, tail, mode):
        h_x, h_y, h_z = torch.chunk(head, 3, dim=2)
        r_theta, r_z  = torch.chunk(relation, 2, dim=2)
        t_x, t_y, t_z = torch.chunk(tail, 3, dim=2)

        phase_relation = r_theta / (self.embedding_range.item() / pi)
        cos_relation = torch.cos(phase_relation)
        sin_relation = torch.sin(phase_relation)

        h_r_x = h_x * cos_relation - h_y * sin_relation
        h_r_y = h_x * sin_relation + h_y * cos_relation
        h_r_z = h_z + r_z - t_z

        score = torch.stack([h_r_x, h_r_y, h_r_z], dim=0)
        score = torch.norm(score, dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    # 双周期
    def periodic2(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        head = head / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)

        phase = head + relation -tail

        phase1, phase2 = torch.chunk(phase, 2, dim=2)
        score = torch.sin(phase1) * torch.sin(phase2)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2)*self.modulus
        return score


    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        # pytorch中，启用 batch_normalization 和 dropout
        model.train()

        optimizer.zero_grad()

        # 按batch分配
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        # 这里数据都是batch了
        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            # detach() 函数起到了阻断backpropogation的作用
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        # mode = 'single'
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        # 这里的weight和self-adversarial 没有任何联系
        #只不过是一种求负样本loss平均的策略，那就得参考每个样本的重要性了，也就是 subsampling_weight
        # 这个weight来源于word2vec的subsampling weight，
        # 这里是在一个batch中，评估每一个样本的权重
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            # 作者出错了，两者是不同的，完全不同，正如原函数说明里面强调： average precision
            # is different from computing the area under the precision-recall curve with the trapezoidal rule
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)
                        # descending=True 降序排列，得分较高的，排序较为靠前; argsort是按照index编号进行的排序过程
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics

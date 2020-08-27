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
        # self.embedding_range = nn.Parameter(
        #              torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
        #              requires_grad=False
        #          )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # self.embedding_range = nn.Parameter(torch.Tensor([3.14]))
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        if model_name == 'OpticalE_weight':
            self.relation_dim = hidden_dim*2+1
        if model_name == 'OpticalE_dir' or model_name == 'HopticalE_twoamp':
            self.entity_dim = hidden_dim * 3 if double_entity_embedding else hidden_dim
        if model_name == 'OpticalE_2unit' or model_name == 'rOpticalE_2unit':
            self.relation_dim = hidden_dim * 2
        if model_name=='HAKE_one' or model_name=='HopticalE_one':
            self.relation_dim = hidden_dim + 1
        
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
        if model_name=='HAKE_one' or model_name=='HopticalE_one':
            nn.init.ones_(
              tensor=self.relation_embedding[:, 0]
            )
        
        if model_name == 'pRotatE' or model_name == 'rOpticalE_mult' or model_name == 'OpticalE_symmetric' or \
                model_name == 'OpticalE_dir_ampone' or model_name=='OpticalE_interference_term' or model_name=='regOpticalE'\
                or model_name=='regOpticalE_r' or model_name=='HAKE' or model_name=='HAKE_one':
            # self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
            self.modulus = nn.Parameter(torch.Tensor([[self.gamma.item() * 0.5 / self.hidden_dim]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'OpticalE', 'rOpticalE', \
                              'OpticalE_amp', 'OpticalE_dir', 'pOpticalE_dir', 'OpticalE_2unit', 'rOpticalE_2unit',\
                              'OpticalE_onedir', 'OpticalE_weight', 'OpticalE_mult', 'rOpticalE_mult', 'functan',\
                              'Rotate_double', 'Rotate_double_test', 'OpticalE_symmetric', 'OpticalE_polarization', 'OpticalE_dir_ampone', 'OpticalE_relevant_ampone',\
                              'OpticalE_intefere', 'OpticalE_interference_term', 'HopticalE', 'HopticalE_re', 'regOpticalE', 'regOpticalE_r', 'HAKE', 'HAKE_one', 'HopticalE_one']:
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
            'OpticalE_amp': self.OpticalE_amp,
            'OpticalE_dir': self.OpticalE_dir,
            'pOpticalE_dir': self.pOpticalE_dir,
            'OpticalE_2unit': self.OpticalE_2unit,
            'rOpticalE_2unit': self.rOpticalE_2unit,
            'OpticalE_onedir': self.OpticalE_onedir,
            'OpticalE_weight': self.OpticalE_weight,
            'OpticalE_mult': self.OpticalE_mult,
            'rOpticalE_mult': self.rOpticalE_mult,
            'functan': self.functan,
            'Rotate_double': self.Rotate_double,
            'Rotate_double_test': self.Rotate_double_test,
            'OpticalE_symmetric': self.OpticalE_symmetric,
            'OpticalE_polarization': self.OpticalE_polarization,
            'OpticalE_dir_ampone': self.OpticalE_dir_ampone,
            'OpticalE_relevant_ampone': self.OpticalE_relevant_ampone,
            'OpticalE_intefere': self.OpticalE_intefere,
            'OpticalE_interference_term': self.OpticalE_interference_term,
            'HopticalE': self.HopticalE,
            'HopticalE_re': self.HopticalE_re,
            'regOpticalE': self.regOpticalE,
            'regOpticalE_r': self.regOpticalE_r,
            'HopticalE_add': self.HopticalE_add,
            'HAKE': self.HAKE,
            'HAKE_one': self.HAKE_one,
            'HopticalE_one': self.HopticalE_one

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

    def OpticalE_weight(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        bias_relation, relation = relation[:,:,0], relation[:,:,1:]
        weight_relation, p_relation = torch.chunk(relation, 2, dim=2)

        phase_relation = p_relation / (self.embedding_range.item() / pi)
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
        score = (score * weight_relation).sum(dim=2) + bias_relation
        return score

    def OpticalE_2unit(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        relation_head, relation_tail = torch.chunk(phase_relation, 2, dim=2)
        re_relation_h = torch.cos(relation_head)
        im_relation_h = torch.sin(relation_head)

        re_relation_t = torch.cos(relation_tail)
        im_relation_t = torch.sin(relation_tail)

        # if mode == 'head-batch': # 逆旋转处理
        #     re_score = re_relation * re_tail + im_relation * im_tail
        #     im_score = re_relation * im_tail - im_relation * re_tail
        #     re_score = re_score + re_head
        #     im_score = im_score + im_head
        # else:
        #     re_score = re_head * re_relation - im_head * im_relation
        #     im_score = re_head * im_relation + im_head * re_relation
        #     re_score = re_score + re_tail
        #     im_score = im_score + im_tail

        re_score_head = re_head * re_relation_h - im_head * im_relation_h
        im_score_head = re_head * im_relation_h + im_head * re_relation_h

        re_score_head1, re_score_head2 = torch.chunk(re_score_head, 2, dim=2)
        re_score_head = re_score_head1 + re_score_head2

        im_score_head1, im_score_head2 = torch.chunk(im_score_head, 2, dim=2)
        im_score_head = im_score_head1 + im_score_head2


        re_score_tail = re_tail * re_relation_t - im_tail * im_relation_t
        im_score_tail = re_tail * im_relation_t + im_tail * re_relation_t

        re_score_tail1, re_score_tail2 = torch.chunk(re_score_tail, 2, dim=2)
        re_score_tail = re_score_tail1 + re_score_tail2

        im_score_tail1, im_score_tail2 = torch.chunk(im_score_tail, 2, dim=2)
        im_score_tail = im_score_tail1 + im_score_tail2

        re_score = re_score_head + re_score_tail
        im_score = im_score_head + im_score_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = score.sum(dim=2) - self.gamma.item()
        return score

    def rOpticalE_2unit(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        relation_head, relation_tail = torch.chunk(phase_relation, 2, dim=2)
        re_relation_h = torch.cos(relation_head)
        im_relation_h = torch.sin(relation_head)

        re_relation_t = torch.cos(relation_tail)
        im_relation_t = torch.sin(relation_tail)

        # if mode == 'head-batch': # 逆旋转处理
        #     re_score = re_relation * re_tail + im_relation * im_tail
        #     im_score = re_relation * im_tail - im_relation * re_tail
        #     re_score = re_score + re_head
        #     im_score = im_score + im_head
        # else:
        #     re_score = re_head * re_relation - im_head * im_relation
        #     im_score = re_head * im_relation + im_head * re_relation
        #     re_score = re_score + re_tail
        #     im_score = im_score + im_tail

        re_score_head = re_head * re_relation_h - im_head * im_relation_h
        im_score_head = re_head * im_relation_h + im_head * re_relation_h

        re_score_head1, re_score_head2 = torch.chunk(re_score_head, 2, dim=2)
        re_score_head = re_score_head1 + re_score_head2

        im_score_head1, im_score_head2 = torch.chunk(im_score_head, 2, dim=2)
        im_score_head = im_score_head1 + im_score_head2


        re_score_tail = re_tail * re_relation_t - im_tail * im_relation_t
        im_score_tail = re_tail * im_relation_t + im_tail * re_relation_t

        re_score_tail1, re_score_tail2 = torch.chunk(re_score_tail, 2, dim=2)
        re_score_tail = re_score_tail1 + re_score_tail2

        im_score_tail1, im_score_tail2 = torch.chunk(im_score_tail, 2, dim=2)
        im_score_tail = im_score_tail1 + im_score_tail2

        re_score = re_score_head + re_score_tail
        im_score = im_score_head + im_score_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
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



    def OpticalE_amp(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        amp_head, phase_emb_head = torch.chunk(head, 2, dim=2)
        amp_tail, phase_emb_tail = torch.chunk(tail, 2, dim=2)

        phase_r = relation / (self.embedding_range.item() / pi)
        phase_h = phase_emb_head / (self.embedding_range.item() / pi)
        phase_t = phase_emb_tail / (self.embedding_range.item() / pi)

        # if mode == 'head-batch': # 逆旋转处理
        #     re_score = re_relation * re_tail + im_relation * im_tail
        #     im_score = re_relation * im_tail - im_relation * re_tail
        #     re_score = re_score + re_head
        #     im_score = im_score + im_head
        # else:
        #     re_score = re_head * re_relation - im_head * im_relation
        #     im_score = re_head * im_relation + im_head * re_relation
        #     re_score = re_score + re_tail
        #     im_score = im_score + im_tail

        intensity_h = amp_head ** 2
        intensity_t = amp_tail ** 2

        interference = 2 * amp_head * amp_tail * torch.cos(phase_h + phase_r - phase_t)

        score = intensity_h + intensity_t + interference

        score = score.sum(dim=2) - self.gamma.item()
        return score

    def OpticalE_dir(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        amp_head_x, amp_head_y, phase_emb_head = torch.chunk(head, 3, dim=2)
        amp_tail_x, amp_tail_y, phase_emb_tail = torch.chunk(tail, 3, dim=2)
        amp_relation, phase_relation = torch.chunk(relation, 2, dim=2)

        phase_amp = amp_relation / (self.embedding_range.item() / pi)
        phase_r = phase_relation / (self.embedding_range.item() / pi)
        phase_h = phase_emb_head / (self.embedding_range.item() / pi)
        phase_t = phase_emb_tail / (self.embedding_range.item() / pi)

        amp_x = amp_head_x * torch.cos(phase_amp) - amp_head_y * torch.sin(phase_amp)
        amp_y = amp_head_x * torch.sin(phase_amp) + amp_head_y * torch.cos(phase_amp)


        intensity_h = amp_head_x ** 2 + amp_head_y ** 2
        intensity_t = amp_tail_x ** 2 + amp_tail_y ** 2

        interference = 2 * (amp_x * amp_tail_x + amp_y * amp_tail_y) * torch.cos(phase_h + phase_r - phase_t)

        score = intensity_h + intensity_t + interference

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def OpticalE_dir_ampone(self, head, relation, tail, mode):
        # 震动方向改变，但是强度始终为1
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        head = head / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)

        head_dir, head_phase = torch.chunk(head, 2, dim=2)
        tail_dir, tail_phase = torch.chunk(tail, 2, dim=2)

        intensity = 2 * torch.abs(torch.cos(head_dir - tail_dir)) * torch.cos(head_phase + relation - tail_phase) + 2.0
        score = self.gamma.item() - intensity.sum(dim=2) * 0.006

        return score

    def HopticalE(self, head, relation, tail, mode):

        '''
                pi = 3.14159262358979323846
                head_mod, head_phase = torch.chunk(head, 2, dim=2)
                tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
                rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

                head_phase = head_phase / (self.embedding_range.item() / pi)
                tail_phase = tail_phase / (self.embedding_range.item() / pi)
                rel_phase = rel_phase / (self.embedding_range.item() / pi)

                hr_mod = torch.abs(head_mod * rel_mod)
                hr_phase = head_phase + rel_phase

                E_x = hr_mod * torch.cos(hr_phase) + tail_mod.abs() * torch.cos(tail_phase)
                E_y = hr_mod * torch.sin(hr_phase) + tail_mod.abs() * torch.sin(tail_phase)

                score = torch.stack([E_x, E_y], dim=0)
                score = score.norm(dim=0)
                score = self.gamma.item() - score.sum(dim=2)
                return score
                '''

        pi = 3.14159262358979323846

        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        hr_mod = head_mod * rel_mod
        I = hr_mod ** 2 + tail_mod ** 2 + 2 * (hr_mod * tail_mod) * torch.cos(head_phase + rel_phase - tail_phase)
        score = I.sum(dim=2) - self.gamma.item()
        return score



    def HopticalE_re(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        hr_mod = torch.abs(head_mod * rel_mod)
        I = hr_mod ** 2 + tail_mod ** 2 + 2 * (hr_mod * tail_mod).abs() * torch.cos(head_phase + rel_phase - tail_phase)
        score = I.sum(dim=2) - self.gamma.item()
        return score

    def HopticalE_add(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        hr_mod = head_mod + rel_mod
        I = hr_mod ** 2 + tail_mod ** 2 + 2 * (hr_mod * tail_mod).abs() * torch.cos(head_phase + rel_phase - tail_phase)
        score = self.gamma.item() - I.sum(dim=2)
        return score


    def regOpticalE_r(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        #score = (tail_mod ** 2 + head_mod ** 2 + rel_mod ** 2) + 2 * (head_mod * rel_mod -  head_mod * tail_mod - rel_mod * tail_mod) \
        #        + self.modulus * torch.cos(head_phase + rel_phase - tail_phase).abs()
        hr = head_mod * rel_mod.abs()
        score_r = ((hr - tail_mod) ** 2).sum(dim=2)
        score_p = (hr ** 2 + tail_mod ** 2 - 2 * hr * tail_mod * torch.abs(torch.cos(head_phase + rel_phase - tail_phase))).sum(dim=2)

        score = score_r + self.modulus * score_p


        #score_ModE = (head_mod * r) ** 2 + tail_mod ** 2 - 2 * head_mod * r * tail_mod
        score = self.gamma.item() - score

        return score

    def HAKE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        # score = (tail_mod ** 2 + head_mod ** 2 + rel_mod ** 2) + 2 * (head_mod * rel_mod -  head_mod * tail_mod - rel_mod * tail_mod) \
        #        + self.modulus * torch.cos(head_phase + rel_phase - tail_phase).abs()
        score = (head_mod * rel_mod.abs() - tail_mod).norm(p=2, dim=2) + (
                    self.modulus * torch.cos(head_phase + rel_phase - tail_phase)).norm(p=1, dim=2)

        # score_ModE = (head_mod * r) ** 2 + tail_mod ** 2 - 2 * head_mod * r * tail_mod
        score = self.gamma.item() - score

        return score

    def HAKE_one(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = relation[:, :, 0:1], relation[:, :, 1:]

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        # score = (tail_mod ** 2 + head_mod ** 2 + rel_mod ** 2) + 2 * (head_mod * rel_mod -  head_mod * tail_mod - rel_mod * tail_mod) \
        #        + self.modulus * torch.cos(head_phase + rel_phase - tail_phase).abs()
        score = (head_mod * rel_mod.abs() - tail_mod).norm(p=2, dim=2) + (
                self.modulus * torch.cos(head_phase + rel_phase - tail_phase)).norm(p=1, dim=2)
        score = self.gamma.item() - score

        return score

    def HopticalE_one(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        rel_mod, rel_phase = relation[:, :, 0:1], relation[:, :, 1:]

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        hr_mod = torch.abs(head_mod * rel_mod)
        I = hr_mod ** 2 + tail_mod ** 2 + 2 * (hr_mod * tail_mod).abs() * torch.cos(head_phase + rel_phase - tail_phase)
        score = self.gamma.item() -I.sum(dim=2)
        return score


    def regOpticalE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        head_mod, head_phase = torch.chunk(head, 2, dim=2)
        tail_mod, tail_phase = torch.chunk(tail, 2, dim=2)
        #rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = relation / (self.embedding_range.item() / pi)

        score = tail_mod ** 2 + head_mod ** 2 + 2 * torch.abs(tail_mod * head_mod) * torch.cos(
            head_phase + rel_phase - tail_phase) * 2
        # score_ModE = (head_mod * r) ** 2 + tail_mod ** 2 - 2 * head_mod * r * tail_mod
        score = self.gamma.item() - score.sum(dim=2)

        return score




    def HopticalE_twoamp(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        head_mod, _, head_phase = torch.chunk(head, 3, dim=2)
        _, tail_mod, tail_phase = torch.chunk(tail, 3, dim=2)
        rel_mod, rel_phase = torch.chunk(relation, 2, dim=2)

        head_phase = head_phase / (self.embedding_range.item() / pi)
        tail_phase = tail_phase / (self.embedding_range.item() / pi)
        rel_phase = rel_phase / (self.embedding_range.item() / pi)

        hr_mod = head_mod * rel_mod.abs()
        I = hr_mod ** 2 + tail_mod ** 2 + 2 * (hr_mod * tail_mod) * torch.cos(head_phase + rel_phase - tail_phase).abs()
        score = self.gamma.item() - I.sum(dim=2)
        return score


    def OpticalE_interference_term(self, head, relation, tail, mode):
        # 震动方向改变，但是强度始终为1
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        head = head / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)

        head_dir, head_phase = torch.chunk(head, 2, dim=2)
        tail_dir, tail_phase = torch.chunk(tail, 2, dim=2)

        intensity = torch.abs(torch.cos(head_dir - tail_dir) * torch.cos(head_phase + relation - tail_phase))

        score = self.gamma.item() - intensity.sum(dim=2) * self.modulus

        return score

    def OpticalE_pos(self, head, relation, tail, mode):
        # 震动方向改变，但是强度始终为1
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        head = head / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)

        head_dir, head_phase = torch.chunk(head, 2, dim=2)
        tail_dir, tail_phase = torch.chunk(tail, 2, dim=2)

        intensity = 2 * torch.abs(torch.cos(head_dir - tail_dir)) * torch.cos(head_phase + relation - tail_phase) + 2.0

        score = self.gamma.item() - intensity.sum(dim=2) * self.modulus
        return score

    def OpticalE_neg_relevant(self, head, relation, tail, mode):
        # 震动方向改变，但是强度始终为1
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        head = head / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)

        head_dir, head_phase = torch.chunk(head, 2, dim=2)
        tail_dir, tail_phase = torch.chunk(tail, 2, dim=2)

        intensity = 2 * torch.cos(head_phase + relation - tail_phase) + 2.0

        score = self.gamma.item() - intensity.sum(dim=2) * self.modulus
        return score

    def OpticalE_neg_unrelevant(self, head, relation, tail, mode):
        # 震动方向改变，但是强度始终为1
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        head = head / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)

        head_dir, head_phase = torch.chunk(head, 2, dim=2)
        tail_dir, tail_phase = torch.chunk(tail, 2, dim=2)

        intensity = 2 * torch.abs(torch.cos(head_dir - tail_dir)) + 2.0

        score = self.gamma.item() - intensity.sum(dim=2) * self.modulus
        return score


    def OpticalE_relevant_ampone(self, head, relation, tail, mode):

        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        head = head / (self.embedding_range.item() / pi)
        tail = tail / (self.embedding_range.item() / pi)
        relation = relation / (self.embedding_range.item() / pi)


        intensity = 2 * torch.abs(torch.sin(head - tail)) * torch.cos(head + relation - tail) + 2.0

        score = self.gamma.item() - intensity.sum(dim=2) * self.modulus
        return score

    def OpticalE_intefere(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_head = head / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        score = 2 + 2 * torch.cos(phase_head + phase_relation - phase_tail) + torch.abs(torch.cos(phase_head - phase_tail)) * 0.1
        # score = 2 + 2 * torch.cos(phase_head + phase_relation - phase_tail)
        score = self.gamma.item() - score.sum(dim=2)*self.modulus
        return score

    def OpticalE_onedir(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        direction_head, phase_emb_head = torch.chunk(head, 2, dim=2)
        direction_tail, phase_emb_tail = torch.chunk(tail, 2, dim=2)

        phase_r = relation / (self.embedding_range.item() / pi)
        phase_h = phase_emb_head / (self.embedding_range.item() / pi)
        phase_t = phase_emb_tail / (self.embedding_range.item() / pi)

        intensity_h = direction_head ** 2
        intensity_t = direction_tail ** 2

        interference = 2 * (direction_head * direction_tail).sum(dim=2, keepdims=True)\
                       * torch.cos(phase_h + phase_r - phase_t)

        score = (intensity_h + intensity_t + interference).sqrt()

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pOpticalE_dir(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        amp_phase_h, phase_emb_head = torch.chunk(head, 2, dim=2)
        amp_phase_t, phase_emb_tail = torch.chunk(tail, 2, dim=2)


        amp_phase_head = amp_phase_h / (self.embedding_range.item() / pi)
        amp_phase_tail = amp_phase_t / (self.embedding_range.item() / pi)

        phase_r = relation / (self.embedding_range.item() / pi)
        phase_h = phase_emb_head / (self.embedding_range.item() / pi)
        phase_t = phase_emb_tail / (self.embedding_range.item() / pi)

        interference = 2 * self.modulus * torch.cos(amp_phase_head - amp_phase_tail) * torch.cos(phase_h + phase_r - phase_t)

        score = 2 * self.modulus + interference

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def OpticalE_mult(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        # re_relation, im_relation [16, 1, 20]
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score * re_tail
        im_score = im_score * im_tail


        score = re_score + im_score
        score = score.sum(dim=2) - self.gamma.item()
        return score

    def rOpticalE_mult(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_head = relation / (self.embedding_range.item() / pi)
        phase_tail = relation / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)

        phase = phase_head + phase_relation + phase_tail

        # re_relation, im_relation [16, 1, 20]
        # re_relation = torch.cos(phase_relation)
        # im_relation = torch.sin(phase_relation)

        # re_score = re_head * re_relation - im_head * im_relation
        # im_score = re_head * im_relation + im_head * re_relation
        # re_score = re_score * re_tail
        # im_score = im_score * im_tail


        score = self.modulus * self.modulus * torch.cos(phase)
        score = score.sum(dim=2) - self.gamma.item()
        return score
    def functan(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        phase_head = head / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        score = torch.cos(phase_head + phase_relation - phase_tail)
        score = score.sum(dim=2) - self.gamma.item()
        return score

    def Rotate_double(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        # chunk函数是切块用，chunk（tensor，n份，切块的维度），返回tensor的list
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        # relation1, relation2 = torch.chunk(relation, 2, dim=2)
        # re_haed, im_head [16,1,20]; re_tail, im_head [16,2,20]
        # Make phases of relations uniformly distributed in [-pi, pi]

        # phase_relation 属于 正负pi
        phase_relation1 = relation / (self.embedding_range.item() / pi)
        # phase_relation2 = relation2 / (self.embedding_range.item() / pi)
        # re_relation, im_relation [16, 1, 20]
        re_relation1 = torch.cos(phase_relation1)
        im_relation1 = torch.sin(phase_relation1)

        # re_score im_score [16,1,20]; re_tail im_tail [16,2,20]
        re_score = re_head * re_relation1 - im_head * im_relation1
        im_score = re_head * im_relation1 + im_head * re_relation1

        _, _, z = im_score.shape
        im_score_last = im_score[:, :, z-1:]
        im_score_first = im_score[:, :, :z-1]
        im_score = torch.cat([im_score_last, im_score_first], dim=2)

        im_score2 = im_score * re_relation1 - re_score * im_relation1
        re_score2 = im_score * re_relation1 + re_score * im_relation1

        im_score_last = im_score2[:, :, 1:]
        im_score_first = im_score2[:, :, :1]
        im_score2 = torch.cat([im_score_last, im_score_first], dim=2)


        re_score = re_score2 - re_tail
        im_score = im_score2 - im_tail

        # re_score 会 broadcast 成 [16,2,20]
        score = torch.stack([re_score, im_score], dim=0)
        # score [2,16,2,20]
        # tensor.norm() 求范数；默认是2; 得到的结果往往会删除dim=k的那一维
        score = score.norm(dim=0)
        # score [16,2,20]

        # 注意，作者将embedding的每一个维度的距离求和，这个和是1范式的，而上面的距离又是二范式的norm
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def Rotate_double_test(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        # chunk函数是切块用，chunk（tensor，n份，切块的维度），返回tensor的list
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        relation1, relation2 = torch.chunk(relation, 2, dim=2)
        # re_haed, im_head [16,1,20]; re_tail, im_head [16,2,20]
        # Make phases of relations uniformly distributed in [-pi, pi]

        # phase_relation 属于 正负pi
        phase_relation1 = relation1 / (self.embedding_range.item() / pi)
        phase_relation2 = relation2 / (self.embedding_range.item() / pi)
        # re_relation, im_relation [16, 1, 20]
        re_relation1 = torch.cos(phase_relation1)
        im_relation1 = torch.sin(phase_relation1)

        # re_score im_score [16,1,20]; re_tail im_tail [16,2,20]
        re_score = re_head * re_relation1 - im_head * im_relation1
        im_score = re_head * im_relation1 + im_head * re_relation1

        _, _, z = im_score.shape
        im_score_last = im_score[:, :, z-1:]
        im_score_first = im_score[:, :, :z-1]
        im_score = torch.cat([im_score_last, im_score_first], dim=2)

        re_relation2 = torch.cos(phase_relation2)
        im_relation2 = torch.sin(phase_relation2)

        im_score2 = im_score * re_relation2 - re_score * im_relation2
        re_score2 = im_score * re_relation2 + re_score * im_relation2

        im_score_last = im_score2[:, :, 1:]
        im_score_first = im_score2[:, :, :1]
        im_score2 = torch.cat([im_score_last, im_score_first], dim=2)


        re_score = re_score2 - re_tail
        im_score = im_score2 - im_tail

        # re_score 会 broadcast 成 [16,2,20]
        # score = torch.stack([re_score, im_score], dim=0)
        score = torch.cat([re_score, im_score], dim=2)
        # score [2,16,2,20]
        # tensor.norm() 求范数；默认是L2; 得到的结果往往会删除dim=k的那一维
        # score = score.norm(dim=0)
        # score [16,2,20]

        # 注意，作者将embedding的每一个维度的距离求和，这个和是1范式的，而上面的距离又是二范式的norm
        score = score.norm(dim=2) - self.gamma.item()
        return score

    def OpticalE_symmetric(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_head = head / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        # re_relation, im_relation [16, 1, 20]

        score = torch.cos(phase_head + phase_tail - 2*phase_relation)
        score = torch.abs(score)

        score = 10.0 - score.sum(dim=2)
        return score

    def OpticalE_polarization(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # re_haed, im_head [16,1,20]; re_tail, im_tail [16,2,20]
        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_head = head / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)
        delay = relation / (self.embedding_range.item() / pi)

        E_x_re = torch.cos(phase_head)
        E_y_re = torch.sin(phase_head)
        E = torch.stack([E_x_re, E_y_re], dim=3).unsqueeze(dim=4)


        plate_re = torch.cos(delay)
        plate_im = torch.sin(delay)
        zeros = torch.zeros(delay.shape).cuda()
        ones = torch.ones(delay.shape).cuda()
        plate_re = torch.stack([ones,zeros,zeros,plate_re], dim=3).unsqueeze(dim=4).reshape(delay.shape+(2,2))
        plate_im = torch.stack([zeros, zeros, zeros, plate_im], dim=3).unsqueeze(dim=4).reshape(delay.shape+(2,2))

        a = torch.cos(phase_tail)
        b = torch.sin(phase_tail)
        polarizer = torch.stack([a*a, a*b, a*b, a*a], dim=3).unsqueeze(dim=4).reshape(phase_tail.shape+(2,2))


        E_re = polarizer.matmul(plate_re).matmul(E).squeeze(dim=4)
        E_im = polarizer.matmul(plate_im).matmul(E).squeeze(dim=4)

        score = torch.cat([E_re, E_im],dim=3).norm(dim=3)

        score = 12.0 - score.sum(dim=2)
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



        # negative_score = model((positive_sample, negative_sample), mode=mode)
        # # print(negative_score)
        # negative_score1 = torch.sigmoid(negative_score)
        # zeros = torch.zeros_like(negative_score)
        #
        # negative_score2 = torch.where(negative_score1 > 0.9999, zeros, negative_score1)
        #
        #
        # if args.negative_adversarial_sampling:
        #     #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        #     # detach() 函数起到了阻断backpropogation的作用
        #     negative_score = (F.softmax(negative_score2 * args.adversarial_temperature, dim = 1).detach()
        #                       * torch.log(1.0 - negative_score2)).sum(dim = 1)
        #
        # else:
        #     negative_score = torch.log(1.0 - negative_score2).mean(dim = 1)
        #
        #
        # # mode = 'single'
        # positive_score = model(positive_sample)
        # positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        # print(negative_score)
        thre = 400.0
        negative_score1 = torch.where(negative_score > thre, -negative_score, negative_score)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            # detach() 函数起到了阻断backpropogation的作用
            negative_score = (F.softmax(negative_score1 * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(- negative_score1)).sum(dim=1)

        else:
            negative_score = F.logsigmoid(- negative_score1).mean(dim=1)

        # mode = 'single'

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

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
                        # score = torch.sigmoid(score)
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

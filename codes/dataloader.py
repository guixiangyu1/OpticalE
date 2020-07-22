#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch-test':
            # 负例生成；filter_bias 是用来对评分进行综合的；对其他正例进行去除，且得分要-1，对负例的得分不做操作；
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            #将测试的head也标记为0
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch-test':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        # 这个while true用的很精髓，数据会源源不断
        while True:
            for data in dataloader:
                yield data


class RelationIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.rel_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def rel_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        # 这个while true用的很精髓，数据会源源不断
        while True:
            for data in dataloader:
                yield data

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        # true_tail用来记录(h, r) 对应的正确的 t ， 属于diction, tail 记录于 np.array
        self.true_head, self.true_tail, self.true_relation, self.relevant_entities = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    # __getitem__ 就是让实例能跟字典一样取元素: P[idx]
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        # 将常出现的组合，权重反而降低；稀有的组合，反而权重更高。这种模式是模仿word2vec负采样中的 subsampling weight。
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0


        while negative_sample_size <= self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.relevant_entities[tail],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.relevant_entities[head],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample_unrelevant = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample_unrelevant = torch.from_numpy(negative_sample_unrelevant)

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size <= self.negative_sample_size:
            negative_sample = np.random.randint(self.nrelation, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.true_relation[(head, tail)],
                assume_unique=True,
                invert=True)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample_relevant = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample_relevant = torch.from_numpy(negative_sample_relevant)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample_unrelevant, negative_sample_relevant, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        # merges a list of samples to form a mini-batch
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample_unrelevant = torch.stack([_[1] for _ in data], dim=0)
        negative_sample_relevant   = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        # 一个batch中，只有一个mode，因此只取其一
        mode = data[0][4]
        return positive_sample, negative_sample_unrelevant, negative_sample_relevant, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1

            if (head, tail) not in count:
                count[(head, tail)] = start
            else:
                count[(head, tail)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}
        true_relation = {}
        relevant_entities = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)
            if (head, tail) not in true_relation:
                true_relation[(head, tail)] = []
            true_relation[(head, tail)].append(relation)

            if head not in relevant_entities:
                relevant_entities[head] = []
            relevant_entities[head].append(tail)
            if tail not in relevant_entities:
                relevant_entities[tail] = []
            relevant_entities[tail].append(head)


        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
        for head, tail in true_relation:
            true_relation[(head, tail)] = np.array(list(set(true_relation[(head, tail)])))
        for entity in relevant_entities:
            relevant_entities[entity] = np.array(list(set(relevant_entities[entity])))

        return true_head, true_tail, true_relation, relevant_entities

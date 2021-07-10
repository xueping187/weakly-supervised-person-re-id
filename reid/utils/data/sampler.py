from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, ids, train_classlist=None, batch_size=3):
        self.batch_size = batch_size
        self.n_similar = 1
        self.is_training = True
        self.classwiseidx = []
        self.trainidx = list(range(len(ids)))
#        self.classlist = list(range(625)) 
        self.classlist = train_classlist
        self.labels = ids
        self.classwise_mapping()
        
    def classwise_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category:
                        idx.append(i); break;
            self.classwiseidx.append(idx)

    def __len__(self):
        return self.batch_size*30

    def __iter__(self):
        if self.is_training:
            idx_temp = []
            for i in range(30):
                idx = []
                # Load similar pairs
                rand_classid = np.random.choice(len(self.classwiseidx), size=self.n_similar)
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
                    idx.append(self.classwiseidx[rid][rand_sampleid[0]])
                    idx.append(self.classwiseidx[rid][rand_sampleid[1]])
    
                # Load rest pairs
                rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size-2*self.n_similar)
                for r in rand_sampleid:
                    idx.append(self.trainidx[r])
                idx_temp.extend(idx)
          
            return iter(idx_temp)

	

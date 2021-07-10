from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
#from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter



def MIL(element_logits, seq_len, batch_size, labels):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over, 
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''
#    labels = labels.type(torch.cuda.FloatTensor)
    k = np.ceil(seq_len/4).astype('int32')
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).cuda()
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CPAL(x, element_logits, seq_len, n_similar, labels):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    sim_loss = 0.
    n_tmp = 0.
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).cuda()
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).cuda()
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*Variable(labels[i,:])*Variable(labels[i+1,:]))
        n_tmp = n_tmp + torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))
    sim_loss = sim_loss / n_tmp
    return sim_loss


class BaseTrainer(object):
    def __init__(self, model, fixed_layer=True):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.fixed_layer = fixed_layer

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        if self.fixed_layer:
            # The following code is used to keep the BN on the first three block fixed 
            fixed_bns = []
            for idx, (name, module) in enumerate(self.model.module.named_modules()):
                if name.find("layer3") != -1:
                    assert len(fixed_bns) == 22
                    break
                if name.find("bn") != -1:
                    fixed_bns.append(name)
                    module.eval() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        millosses = AverageMeter()
        cpalosses = AverageMeter()	

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, milloss, cpaloss = self._forward(inputs, targets)
#            loss = self._forward(inputs, targets)
    
            losses.update(loss.item(), targets.size(0))
            millosses.update(milloss, targets.size(0))
            cpalosses.update(cpaloss, targets.size(0))

            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.75)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'MILLoss {:.3f} ({:.3f})\t'
                      'CPALoss {:.3f} ({:.3f})\t'                      
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg,
                              millosses.val, millosses.avg,
                              cpalosses.val, cpalosses.avg))
              

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, pids = inputs
        pids = pids.float()
        inputs = Variable(imgs, requires_grad=False)
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
#        seq_len = np.sum(np.max(np.abs(inputs),axis=2)>0,axis=1)
#        seq_len = np.ones((inputs.shape[0],1))*inputs.shape[1]
        seq_len = np.array([100]*10)
        seq_len = np.array(seq_len)
        Lambda = 0.5
        num_similar = 3
        self.model.eval()
        final_features,element_logits = self.model(inputs)        
        
        milloss = MILL(element_logits, seq_len, inputs.shape[0], targets)
        cpaloss = CPAL(final_features, element_logits, seq_len, num_similar, targets)    
        total_loss = Lambda * milloss + (1-Lambda) * cpaloss        
        return total_loss, milloss, cpaloss



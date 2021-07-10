
from __future__ import print_function, absolute_import
import sys
sys.path.append('/home/123/Downloads/dataset')

from reid.bottom_up import *
from reid import datasets
from reid import models
from reid.trainers import Trainer
import numpy as np
import argparse
import os, sys, time
from reid.utils.logging import Logger
import os.path as osp
from torch.backends import cudnn
from torch import nn
import torch
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.serialization import save_checkpoint, load_checkpoint
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
#from reid.weak_evaluation import testing
from reid.evaluators import Evaluator

def get_dataloader(dataset, ids, train_classlist=None, training=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if training:
        transformer = T.Compose([
            T.RandomSizedRectCrop(256, 128),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])
        batch_size = 10
        data_loader = DataLoader(
            Preprocessor(dataset, root='/home/123/Downloads/data/DukeMTMC-VideoReID/images', 
                         num_samples=100,
                         transform=transformer, is_training=training, max_frames=1600),
            batch_size=batch_size, num_workers=8,
            sampler = RandomIdentitySampler(ids, train_classlist, batch_size = batch_size),
            pin_memory=True, drop_last=training)
    
        current_status = "Training" if training else "Testing"
        print("Create dataloader for {} with batch_size {}".format(current_status, batch_size))
    else:
        transformer = T.Compose([
            T.RectScale(256, 128),
            T.ToTensor(),
            normalizer,
        ])
        batch_size = 1
        data_loader = DataLoader(
            Preprocessor(dataset, root='/home/123/Downloads/data/DukeMTMC-VideoReID/images', 
                         num_samples=240,
                         transform=transformer, is_training=training, max_frames=1000),
            batch_size=batch_size, num_workers=8,
            shuffle=training, pin_memory=True, drop_last=training)
    
        current_status = "Training" if training else "Testing"
        print("Create dataloader for {} with batch_size {}".format(current_status, batch_size))
    return data_loader

def train(model, dataset_all, dropout=0.5):

#    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    train_data = dataset_all.train
    train_ids = dataset_all.train_ids
    train_classlist = dataset_all.train_classlist
    query_data = dataset_all.query
    query_ids = dataset_all.query_ids
#    query_ids = list(set(range(1000,1634))-set([1005,1006,1052])) # MARS dataset
    gallery_data = dataset_all.gallery
    gallery_ids = dataset_all.gallery_ids

    # adjust training epochs and learning rate
    epochs = 40
    init_lr = 0.01 
    step_size = 6
#    device = torch.device("cuda")
    """ create model and dataloader """
    train_dataloader = get_dataloader(train_data, train_ids, train_classlist, training=True)
    query_dataloader = get_dataloader(query_data, query_ids, training=False)
    gallery_dataloader = get_dataloader(gallery_data, gallery_ids, training=False)
    # the base parameters for the backbone (e.g. ResNet50)
    base_param_ids = set(map(id, model.module.CNN.base.parameters()))

    # we fixed the first three blocks to save GPU memory
    base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.CNN.base.parameters())

    # params of the new layers
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]

    # set the learning rate for backbone to be 0.1 times
    param_groups = [
        {'params': base_params_need_for_grad, 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # change the learning rate by step
    def adjust_lr(epoch, step_size):
        lr = init_lr / (10 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    """ main training process """
    trainer = Trainer(model, fixed_layer=True)
    evaluator = Evaluator(model)
#    print('Test with the original model: ')
#    top1 = evaluator.evaluate(query_dataloader,gallery_dataloader,query_ids,gallery_ids)
    for epoch in range(epochs):
        adjust_lr(epoch, step_size)
        trainer.train(epoch, train_dataloader, optimizer, print_freq=max(5, len(train_dataloader) // 10 * 10))
        if (epoch % 10 ==0) & (epoch != 0):
#            testing(model,query_data,query_ids,gallery_data,gallery_ids)
            top1 = evaluator.evaluate(query_dataloader,gallery_dataloader,query_ids,gallery_ids)
    return top1

def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True

#    save_path = args.logs_dir
    sys.stdout = Logger(osp.join(args.logs_dir, 'log_weak_duke_5'+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))

    # get all unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))

    model = models.create(args.arch, dropout=0.5,embeding_fea_size=2048, 
                          num_classes=702,fixed_layer=True)
    model = nn.DataParallel(model).cuda()
    
    top1 = train(model, dataset_all)
    print(top1)
    save_checkpoint({
            'state-dict':model.module.state_dict()},is_best=0,fpath=osp.join(args.logs_dir,'checkpoint1.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='weak person re-id')
    parser.add_argument('-d', '--dataset', type=str, default='DukeMTMC-VideoReID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-f', '--fea', type=int, default=2048)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'new_logs'))
    parser.add_argument('--max_frames', type=int, default=900)
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    main(parser.parse_args())


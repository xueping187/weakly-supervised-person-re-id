from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
import numpy as np
from scipy.spatial.distance import cdist
from torch.backends import cudnn

def extract_features(model, data_loader):
    model.eval()
    features = []
    print("Begin to extract features...")
    for k, imgs in enumerate(data_loader):
        inputs, ids = imgs
        fcs, _ = extract_cnn_feature(model, inputs)
        fnorm = torch.norm(fcs,p=2,dim=1,keepdim=True)
        out = fcs.div(fnorm.expand_as(fcs))
#        features += _fcs
        out = out[0].numpy()
#        print('sample:{0:2d}, shape:({1:4d},{2: 4d})'.format(k,out.shape[0],out.shape[1]))
        features.append(out)

#v        for fname, fc, pool5, pid in zip(fnames, _fcs, pool5s, pids):
#            features[fname] = pool5
#            fcs[fname] = fc
#            labels[fname] = pid

#    cudnn.benchmark = True
    return features


def pairwise_distance(query_feats, gallery_feats, metric=None):
    dist = np.zeros((len(query_feats),len(gallery_feats)))
    avg_query_fea = []
    for fea in query_feats:
#        fea = fea.numpy()
        temp = np.mean(fea, axis=0)[np.newaxis,:]
        avg_query_fea.append(temp)
#    for fea in gallery_feats:
##        fea = fea.numpy()
#        gallery_numpy_feats.append(fea)
        
    for i in range(len(query_feats)):
        for j in range(len(gallery_feats)):
            d = np.min(cdist(avg_query_fea[i],gallery_feats[j]))
            dist[i][j]=d

#    x = torch.cat([features["".join(f)].unsqueeze(0) for f, _, _, _ in query], 0)
#    y = torch.cat([features["".join(f)].unsqueeze(0) for f, _, _, _ in gallery], 0)
#    m, n = x.size(0), y.size(0)
#    x = x.view(m, -1)
#    y = y.view(n, -1)
#    if metric is not None:
#        x = metric.transform(x)
#        y = metric.transform(y)
#        x = torch.from_numpy(x)
#        y = torch.from_numpy(y)
#    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#    dist.addmm_(1, -2, x, y.t())
    return dist

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20, 50)):
        
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)        

    # Compute all kinds of CMC scores
    cmc_configs = {
        'mars': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, 
                      cmc_scores['mars'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['mars'][0]

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
    def evaluate(self, query_data_loader, gallery_data_loader, query_ids, gallery_ids, metric=None):
        query_features = extract_features(self.model, query_data_loader)
#        print(len(query_features))
#        np.save('queryppp.npy',query_features)
        gallery_features = extract_features(self.model,gallery_data_loader)
        distmat = pairwise_distance(query_features, gallery_features, metric=metric)
#        np.save('galleryyy.npy',gallery_features)
        return evaluate_all(distmat, query_ids=query_ids, gallery_ids=gallery_ids)

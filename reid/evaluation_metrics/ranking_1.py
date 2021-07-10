from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics.base import _average_binary_score
from sklearn.metrics import precision_recall_curve, auc
# from sklearn.metrics import average_precision_score


#from ..utils import to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def average_precision_score(y_true, y_score, average="macro",
                            sample_weight=None):
    def _binary_average_precision(y_true, y_score, sample_weight=None):
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
        return auc(recall, precision)

    return _average_binary_score(_binary_average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
#    distmat = to_numpy(distmat)
    m, n = distmat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    indices = np.argsort(distmat, axis=1)
    gallery_ids = gallery_ids[indices]

    matches = []
    for i in range(len(query_ids)):
        for g in gallery_ids[i]:
            matches.append(query_ids[i] in g)    
    matches = np.reshape(matches,(m,n)) 
    np.save('matches.npy',matches)
    np.save('distmat.npy',distmat)
    np.save('gallery_ids.npy',gallery_ids)
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = np.ones((n),dtype=bool)
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    print(num_valid_queries)
    np.save('ret.npy',ret)        
    return ret.cumsum() / num_valid_queries

def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    m, n = distmat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    indices = np.argsort(distmat, axis=1)
    gallery_ids = gallery_ids[indices]
    matches = []
    for i in range(len(query_ids)):
        for g in gallery_ids[i]:
            matches.append(query_ids[i] in g)    
    matches = np.reshape(matches,(m,n)) 
    
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = np.ones((n),dtype=bool)
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
        np.save('aps.npy',aps)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)
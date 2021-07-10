from __future__ import print_function
import os.path as osp
import random
import numpy as np
# import weak_labels
from ..serialization import read_json
from ..weak_labels import weak_labels


def _pluck(identities, indices, query_ids=None, training=False, part = 'gallery'):
    
    if training:
        data = []
        labels = []
        classlist = []
        cam_0 = []
        cam_5 = []
        cam_1 = []
        cam_2 = []
        cam_3 = []
        cam_4 = []
        label_0 = []
        label_1 = []
        label_2 = []
        label_3 = []
        label_4 = []
        label_5 = []        
        for index, pid in enumerate(indices):
            pid_images = identities[pid]
            classlist.append(pid)
            for camid, video_ids in enumerate(pid_images):
                for video_id in video_ids:
                    images = video_ids[video_id]
                    if len(images) > 30:
                        images = random.sample(images, 30)                    
                    if camid == 0:
                        cam_0.append(images)
                        label_0.append(pid)
                    elif camid == 1:
                        cam_1.append(images)
                        label_1.append(pid)                    
                    elif camid == 2:
                        cam_2.append(images)
                        label_2.append(pid)                    
                    elif camid == 3:
                        cam_3.append(images)
                        label_3.append(pid)                    
                    elif camid == 4:
                        cam_4.append(images)
                        label_4.append(pid)                    
                    else:
                        cam_5.append(images)
                        label_5.append(pid)                    
                        
        data.append(cam_0), data.append(cam_1), data.append(cam_2), data.append(cam_3)
        data.append(cam_4), data.append(cam_5)
        labels.append(label_0), labels.append(label_1),labels.append(label_2), labels.append(label_3),
        labels.append(label_4), labels.append(label_5)
#        classlist = list(range(625))
            
        trainset, ids = weak_labels(data, labels, classlist)
        return trainset, ids, classlist
    else:
        data = []
        labels = []
        classlist = []
        queryset = []
        cam_0 = []
        cam_5 = []
        cam_1 = []
        cam_2 = []
        cam_3 = []
        cam_4 = []
        label_0 = []
        label_1 = []
        label_2 = []
        label_3 = []
        label_4 = []
        label_5 = []         
        for index, pid in enumerate(indices):
            if pid not in [999,1000,1006,1007,1053]:
                pid_images = identities[pid]
    #            per_query = []
                classlist.append(pid)
                flag = 1
                for camid, video_ids in enumerate(pid_images):
    #                per_query.append(video_ids[0])
                    for video_id in video_ids:
                        images = video_ids[video_id]
                        if flag == 1:
                            queryset.append((tuple(images),pid))
                            flag = 0
                            continue
                        if len(images) > 200:
                            images = random.sample(images, 200)                        
                        if camid == 0:
                            cam_0.append(images)
                            label_0.append(pid)
                        elif camid == 1:
                            cam_1.append(images)
                            label_1.append(pid)                    
                        elif camid == 2:
                            cam_2.append(images)
                            label_2.append(pid)                    
                        elif camid == 3:
                            cam_3.append(images)
                            label_3.append(pid)                    
                        elif camid == 4:
                            cam_4.append(images)
                            label_4.append(pid)                    
                        else:
                            cam_5.append(images)
                            label_5.append(pid)                    
                        
        data.append(cam_0), data.append(cam_1), data.append(cam_2), data.append(cam_3)
        data.append(cam_4), data.append(cam_5)
        labels.append(label_0), labels.append(label_1),labels.append(label_2), labels.append(label_3),
        labels.append(label_4), labels.append(label_5)
            
        galleryset, ids = weak_labels(data, labels, classlist)
        
        return galleryset,ids,queryset,classlist
#        if part == 'gallery':
#            return feats,ids,classlist
#        else:
#            return query,classlist,classlist     


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train,self.train_ids = [], []
        self.train_classlist = []
        
        self.query, self.query_ids = [], [] 
        self.gallery,self.gallery_ids = [], []
        self.num_train_ids = 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        train_pids = np.asarray(self.split['train'])


        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']

        self.train,self.train_ids, self.train_classlist  = _pluck(identities, train_pids, training=True,part='training')
#        self.query,self.query_ids, _ = _pluck(identities, self.split['gallery'], self.split['query'],training=False,part='query')                
        self.gallery,self.gallery_ids, self.query, self.query_ids =\
        _pluck(identities, self.split['gallery'], self.split['query'], training=False)
        self.num_train_ids = len(train_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # tracklets")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.query_ids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.query_ids), len(self.gallery)))
            print()

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))

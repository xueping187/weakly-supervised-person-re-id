# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:03:40 2019

@author: acer
"""
import numpy as np
import random


def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
	return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)

def weak_labels(data,labels,classlist):
    weak_data = []
    weak_ids = []
    for c in range(len(data)):
        vid_temp = []
        num_vids = len(data[c])
        flag = 1
        while(flag):
            samples = list(set(range(num_vids))-set(np.array(vid_temp)))
            dtemp = []
            ltemp = []  
            rand_sampleid = np.random.choice(range(3,6),1)
            if rand_sampleid[0] <= len(samples):          
                vids = np.array(random.sample(samples,rand_sampleid[0]))   
                unique_pids = np.unique(np.array(labels[c])[vids])
                ltemp = strlist2multihot(list(unique_pids),classlist)
                vid_temp = vid_temp + list(vids)
                for i in vids:
                    dtemp = dtemp + data[c][i]
                weak_data.append((tuple(dtemp),ltemp))
                weak_ids.append(unique_pids)                        
            else:
                if(len(samples)==0):
                    flag = 0 
                else:
                    unique_pids = np.unique(np.array(labels[c])[np.array(samples)])
                    ltemp = strlist2multihot(list(unique_pids),classlist)
                    for i in samples:
                        dtemp = dtemp + data[c][i]
                    weak_data.append((tuple(dtemp),ltemp))
                    weak_ids.append(unique_pids)  		
                    flag = 0
    return weak_data, weak_ids                                                             


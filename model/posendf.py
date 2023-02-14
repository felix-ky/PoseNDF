from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
import ipdb
#todo: create a base trainer
from model.network.net_modules import  StructureEncoder, DFNet

import time
import pickle as pkl

class PoseNDF(torch.nn.Module):

    def __init__(self, opt):
        super(PoseNDF, self).__init__()

        self.device = opt['train']['device']

        # create all the models:
        # self.shape_model = ShapeNet().to(self.device)
        # self.pose_model = PoseNet().to(self.device)
        self.enc = None
        if opt['model']['StrEnc']['use']:
            self.enc = StructureEncoder(opt['model']['StrEnc']).to(self.device)

        self.dfnet = DFNet(opt['model']['CanSDF']).to(self.device)
        
        
        #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.loss = opt['train']['loss_type']

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
        
       
    def train(self, mode=True):
        super().train(mode)


    def compute_distance(self, rand_pose):
        """online data generation, not used"""
        rand_pose = rand_pose.unsqueeze(1).repeat(1, len(self.train_poses),1,1)
        train_pose = self.train_poses.unsqueeze(0).repeat( len(rand_pose),1, 1,1)
        dist = torch.sum(torch.arccos(torch.sum(rand_pose*train_pose,dim=3)),dim=2)/2.0  #ToDo: replace with weighted sum, why sqrt??, refer to eq2
        dist_vals = torch.mean(torch.topk(dist,k=5,dim=1,largest=False)[0],dim=1)
        return dist_vals

    def forward(self, inputs ):
        pose = inputs['pose'].to(device=self.device) # (B,5*num_sample,21,4) num_samples是prepare_traindata中的采样数
        # ipdb.set_trace() 
        dist_gt = inputs['dist'].to(device=self.device) # [B,5*num_sample, 5(k_dist)] (2,1000,5)
        rand_pose_in = torch.nn.functional.normalize(pose.to(device=self.device),dim=2)
        if self.enc:
            rand_pose_in = self.enc(rand_pose_in.reshape(pose.shape[0],-1,84)) # (B*5*num_sample, 126)
            # ipdb.set_trace() 
        dist_pred = self.dfnet(rand_pose_in.reshape(-1,126)) #[B,5*num_sample, 1]
        # ipdb.set_trace() 
        loss = self.loss_l1(dist_pred, dist_gt[:,:,0].reshape(-1,1)) # 84的时候需要dist_pred.squeeze()

        # ipdb.set_trace() 
        return loss, {'dist': loss }




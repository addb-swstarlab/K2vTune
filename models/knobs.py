import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WK_NUM = 16

class Knob:
    def __init__(self, knobs, internal_metrics, external_metrics, opt, target_wk, sample_size):
        """
            This class includes knobs, internal metrics, external metrics, target workload info and scaler of data
        """
        self.knobs = knobs
        self.internal_metrics = internal_metrics
        self.external_metrics = external_metrics
        self.opt = opt
        self.target_wk = target_wk
        self.columns = self.knobs.columns
        self.index_value = self.get_index_value()
        self.KNOBSONEHOT_PATH = 'data/knobsOneHot.npy'
        self.knobs_one_hot = self.load_knobsOneHot()
        self.TABLE_PATH = 'data/lookuptable'
        self.DEFAULT_EM_PATH = 'data/external/default_external.csv'
        self.default_trg_em = self.get_trg_default()     
        self.sample_size = sample_size # define sample size, max is 20,000
        self.reduce_sample_size()

    def reduce_sample_size(self):
        if self.sample_size != self.knobs.shape[0]:
            RandomGenerator = np.random.RandomState(seed=1234) # fix random state
            self.rand_idx = RandomGenerator.randint(0, self.knobs.shape[0], self.sample_size)
            self.knobs = self.knobs.iloc[self.rand_idx]
            
            for wk in range(WK_NUM):
                self.internal_metrics[wk] = self.internal_metrics[wk].iloc[self.rand_idx]
                self.external_metrics[wk] = self.external_metrics[wk].iloc[self.rand_idx]
                
            self.knobs_one_hot = self.knobs_one_hot[self.rand_idx]

    def split_data(self, s_wk): # s_wk is similar workload with target workload
        self.s_wk = s_wk
        self.s_internal_metrics = self.internal_metrics[self.s_wk]
        self.s_external_metrics = self.external_metrics[self.s_wk]
        self.X_tr, self.X_te, self.im_tr, self.im_te, self.em_tr, self.em_te, self.knob_tr, self.knob_te = \
            train_test_split(self.knobs_one_hot, self.s_internal_metrics, self.s_external_metrics, self.knobs, test_size=0.2, random_state=22)
        self.X_tr = torch.Tensor(self.X_tr).cuda()
        self.X_te = torch.Tensor(self.X_te).cuda()

    def scale_data(self):
        self.scaler_im = MinMaxScaler().fit(self.im_tr) # [0, 1]
        self.scaler_em = MinMaxScaler().fit(self.em_tr)
        self.scaler_k = MinMaxScaler().fit(self.knob_tr)

        self.norm_im_tr = torch.Tensor(self.scaler_im.transform(self.im_tr)).cuda()
        self.norm_im_te = torch.Tensor(self.scaler_im.transform(self.im_te)).cuda()
        self.norm_em_tr = torch.Tensor(self.scaler_em.transform(self.em_tr)).cuda()
        self.norm_em_te = torch.Tensor(self.scaler_em.transform(self.em_te)).cuda()
        self.norm_k_tr = torch.Tensor(self.scaler_k.transform(self.knob_tr)).cuda()
        self.norm_k_te = torch.Tensor(self.scaler_k.transform(self.knob_te)).cuda()

        # self.default_trg_em = self.scaler_em.transform([self.default_trg_em])[0]
        self.default_trg_em = self.scaler_em.transform(self.default_trg_em)

    def get_trg_default(self):
        '''
            To get default results on target workload, self.target_wk
        '''
        default_em = pd.read_csv(self.DEFAULT_EM_PATH,index_col=0)
        # default_em = default_em.to_numpy()
        return default_em.iloc[self.target_wk:self.target_wk+1, :] # [time, rate, waf, sa]

    def get_index_value(self):
        '''
            To get index value from each knob data
        '''
        self.index_value = dict()
        for col in self.columns:
            iv = self.knobs[[col]].value_counts()#.reset_index(level=0).drop(columns=0)
            iv = iv.sort_index()
            idx = [str(round(_[0], 2)) for _ in list(iv.index)] # For prevent floating point on float value
            self.index_value[col] = pd.Series(data=range(len(iv)), index=idx)
        return self.index_value
    
    def make_knobsOneHot(self, k):
        '''
            make one-hot vector from knob, this step takes much time and so it should run few
        '''
        knobs_one_hot = torch.Tensor()
        for i in range(len(k)):   
            sample = torch.Tensor()
            for col in self.columns:
                try:
                    knob_one_hot = torch.zeros(len(self.index_value[col]))
                    val = str(round(k[col][i], 2)) # For prevent floating point on float value
                    knob_one_hot[self.index_value[col][val]] = 1
                    sample = torch.cat((sample, knob_one_hot))
                ####################################################################################
                except: # get the key with string.... fucking floating point #######################
                    print("val: ", val)
                    print("col: ", col)
                    print("k[col][i]: ", k[col][i])
                    print("index_value key: ", self.index_value[col].keys())
                    print("WHAT? ", self.index_value[col][val])
            sample = sample.unsqueeze(0)
            knobs_one_hot = torch.cat((knobs_one_hot, sample))
        return np.array(knobs_one_hot)
        

    def load_knobsOneHot(self, k=None, save=True):
        '''
            if there is pregained one-hot numpy file, do first command.
            if save is False, this is used at Genetic Algorithm steps
            Otherwise, make one-hot vector and save the vectors to .npy.
        '''
        if self.opt.topK is not None:
            self.KNOBSONEHOT_PATH = f'data/{self.opt.target}_top{self.opt.topK}_knobsOneHot.npy'
        
        if os.path.exists(self.KNOBSONEHOT_PATH) and save: 
            return np.load(self.KNOBSONEHOT_PATH)
        elif not save: # for GA not save gained one-hot vectors
            return self.make_knobsOneHot(k)
        else:
            np.save(self.KNOBSONEHOT_PATH, self.make_knobsOneHot(self.knobs))
            return self.load_knobsOneHot()

    def set_lookuptable(self, table):
        self.lookuptable = table

    def set_knob2vec(self): ## set knob2vec and wrap them to tensor
        self.knob2vec_tr = torch.Tensor(self.get_knob2vec(self.X_tr, self.lookuptable)).cuda()
        self.knob2vec_te = torch.Tensor(self.get_knob2vec(self.X_te, self.lookuptable)).cuda()

    def get_knob2vec(self, data, table):
        k2v = np.zeros((data.shape[0], self.knobs.shape[-1], table.shape[1]))

        for i in range(data.shape[0]):
            if torch.is_tensor(data):
                idx = (data[i]==1).nonzero(as_tuple=False).squeeze().cpu().detach().numpy()
            else:
                idx = (data[i]==1).nonzero().squeeze().cpu().detach().numpy()
            k2v[i] = table[idx]
        return k2v
    
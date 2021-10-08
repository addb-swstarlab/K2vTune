import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Knob:
    def __init__(self, knobs, internal_metrics, external_metrics, target_wk):
        """
            This class includes knobs, internal metrics, external metrics, target workload info and scaler of data
        """
        self.knobs = knobs
        self.internal_metrics = internal_metrics
        self.external_metrics = external_metrics
        self.target_wk = target_wk
        self.columns = self.knobs.columns
        self.index_value = self.get_index_value()
        self.KNOBSONEHOT_PATH = 'data/knobsOneHot.npy'
        self.knobs_one_hot = self.load_knobsOneHot()
        self.TABLE_PATH = 'data/lookuptable'
        self.DEFAULT_EM_PATH = 'data/external/default_external.csv'
        self.default_trg_em = self.get_trg_default()

    def split_data(self, s_wk): # s_wk is similar workload with target workload
        self.s_wk = s_wk
        self.s_internal_metrics = self.internal_metrics[self.s_wk]
        self.s_external_metrics = self.external_metrics[self.s_wk]
        self.X_tr, self.X_te, self.im_tr, self.im_te, self.em_tr, self.em_te, self.knob_tr, self.knob_te = \
            train_test_split(self.knobs_one_hot, self.s_internal_metrics, self.s_external_metrics, self.knobs, test_size=0.2, random_state=22)
        self.X_tr = torch.Tensor(self.X_tr).cuda()
        self.X_te = torch.Tensor(self.X_te).cuda()

    def scale_data(self):
        self.scaler_im = MinMaxScaler().fit(self.im_tr)
        self.scaler_em = MinMaxScaler().fit(self.em_tr)
        self.scaler_k = MinMaxScaler().fit(self.knob_tr)

        self.norm_im_tr = torch.Tensor(self.scaler_im.transform(self.im_tr)).cuda()
        self.norm_im_te = torch.Tensor(self.scaler_im.transform(self.im_te)).cuda()
        self.norm_em_tr = torch.Tensor(self.scaler_em.transform(self.em_tr)).cuda()
        self.norm_em_te = torch.Tensor(self.scaler_em.transform(self.em_te)).cuda()
        self.norm_k_tr = torch.Tensor(self.scaler_k.transform(self.knob_tr)).cuda()
        self.norm_k_te = torch.Tensor(self.scaler_k.transform(self.knob_te)).cuda()

        self.default_trg_em = self.scaler_em.transform([self.default_trg_em])[0]

    def get_trg_default(self):
        default_em = pd.read_csv(self.DEFAULT_EM_PATH,index_col=0)
        default_em = default_em.to_numpy()
        return default_em[self.target_wk] # [time, rate, waf, sa]

    def get_index_value(self):
        self.index_value = dict()
        for col in self.columns:
            iv = self.knobs[[col]].value_counts()#.reset_index(level=0).drop(columns=0)
            iv = iv.sort_index()
            self.index_value[col] = pd.Series(data=range(len(iv)), index=iv.index)
        return self.index_value
    
    def make_knobsOneHot(self, k):
        knobs_one_hot = torch.Tensor()
        for i in range(len(k)):   
            sample = torch.Tensor()
            for col in self.columns:
                knob_one_hot = torch.zeros(len(self.index_value[col]))
                knob_one_hot[self.index_value[col][k[col][i]]] = 1
                sample = torch.cat((sample, knob_one_hot))
            sample = sample.unsqueeze(0)
            knobs_one_hot = torch.cat((knobs_one_hot, sample))
        return np.array(knobs_one_hot)
        

    def load_knobsOneHot(self, k=None, save=True):
        if os.path.exists(self.KNOBSONEHOT_PATH) and save:
            return np.load(self.KNOBSONEHOT_PATH)
        elif not save: # for GA
            return self.make_knobsOneHot(k)
        else:
            np.save(self.KNOBSONEHOT_PATH, self.make_knobsOneHot(self.knobs))
            return self.load_knobsOneHot()

    def set_lookuptable(self, table):
        self.lookuptable = table

    def set_knob2vec(self):
        self.knob2vec_tr = torch.Tensor(self.get_knob2vec(self.X_tr, self.lookuptable)).cuda()
        self.knob2vec_te = torch.Tensor(self.get_knob2vec(self.X_te, self.lookuptable)).cuda()

    def get_knob2vec(self, data, table):
        k2v = np.zeros((data.shape[0], 22, table.shape[1]))
        for i in range(data.shape[0]):
            if torch.is_tensor(data):
                idx = (data[i]==1).nonzero(as_tuple=False).squeeze().cpu().detach().numpy()
            else:
                idx = (data[i]==1).nonzero().squeeze().cpu().detach().numpy()
            k2v[i] = table[idx]
        return k2v
    
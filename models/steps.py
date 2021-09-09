import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from models.network import RocksDBDataset, SingleNet
from models.train import train, valid

def euclidean_distance(a, b):
    res = a - b
    res = res ** 2
    res = np.sqrt(res)
#     return res
    return np.average(res)

def get_euclidean_distance(internal_dict, logger, opt):
    scaler = MinMaxScaler().fit(pd.concat(internal_dict))
    
    wk = []
    for im_d in internal_dict:
        wk.append(scaler.transform(internal_dict[im_d].iloc[:opt.target_size, :]))
        
    big = 100
    for i in range(16):
        ed = euclidean_distance(wk[opt.target], wk[i])
        if ed<big and opt.target != i: 
            big=ed
            idx = i
        logger.info(f'{i}th \t{ed}')
    logger.info(f'best similar workload is {idx}th')

    return idx

def train_knob2vec(knobs, logger, opt):
    Dataset_tr = RocksDBDataset(knobs.X_tr, knobs.norm_im_tr)

    loader_tr = DataLoader(dataset = Dataset_tr, batch_size = 32, shuffle=True)

    model = SingleNet(input_dim=knobs.X_tr.shape[1], hidden_dim=1024, output_dim=knobs.norm_im_tr.shape[-1]).cuda()

    for epoch in range(opt.epochs):
        loss_tr = train(model, loader_tr, opt.lr)
        
        logger.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr}")

    table = model.knob_fc[0].weight.T.cpu().detach().numpy()
    
    if not os.path.exists(knobs.TABLE_PATH):
        os.mkdir(knobs.TABLE_PATH)
    if not os.path.exists(os.path.join(knobs.TABLE_PATH, str(knobs.s_wk))):
        os.mkdir(os.path.join(knobs.TABLE_PATH, str(knobs.s_wk)))
    
    np.save(os.path.join(knobs.TABLE_PATH, str(knobs.s_wk), 'LookupTable.npy'), table)
    
    return table

def train_fitness(knobs, logger, opt):
    if opt.mode == 'dnn':
        Dataset_K2vec_tr = RocksDBDataset(torch.reshape(knobs.knob2vec_tr, (knobs.knob2vec_tr.shape[0], -1)), knobs.norm_em_tr)
        Dataset_K2vec_te = RocksDBDataset(torch.reshape(knobs.knob2vec_te, (knobs.knob2vec_te.shape[0], -1)), knobs.norm_em_te)
    elif opt.mode == 'gru':
        Dataset_K2vec_tr = RocksDBDataset(knobs.knob2vec_tr, knobs.norm_em_tr)
        Dataset_K2vec_te = RocksDBDataset(knobs.knob2vec_te, knobs.norm_em_te)

    loader_K2vec_tr = DataLoader(dataset = Dataset_K2vec_tr, batch_size = 32, shuffle=True)
    loader_K2vec_te = DataLoader(dataset = Dataset_K2vec_te, batch_size = 32, shuffle=False)

    if opt.mode == 'dnn':
        model = SingleNet(input_dim=torch.reshape(knobs.knob2vec_tr, (knobs.knob2vec_tr.shape[0], -1)).shape[-1], hidden_dim=opt.hidden_size, output_dim=knobs.norm_em_tr.shape[-1]).cuda()
    print(knobs.knob2vec_tr.shape[-1], opt.hidden_size, knobs.norm_em_tr.shape[-1])
    for epoch in range(opt.epochs):
        loss_tr = train(model, loader_K2vec_tr, opt.lr)
        loss_te, outputs = valid(model, loader_K2vec_te)

        logger.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")

    return model, outputs

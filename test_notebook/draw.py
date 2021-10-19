import torch
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append('../')
from models.network import *
from models.utils import *

def get_index_value(raw_knobs):
    index_value = dict()
    for col in raw_knobs.columns:
        iv = raw_knobs[[col]].value_counts()#.reset_index(level=0).drop(columns=0)
        iv = iv.sort_index()
        index_value[col] = pd.Series(data=range(len(iv)), index=iv.index)
    return index_value

def make_knobsOneHot(index_value, k, raw_knobs):
        knobs_one_hot = torch.Tensor()
        for i in range(len(k)):   
            sample = torch.Tensor()
            for col in raw_knobs.columns:
                knob_one_hot = torch.zeros(len(index_value[col]))
                knob_one_hot[index_value[col][k[col][i]]] = 1
                sample = torch.cat((sample, knob_one_hot))
            sample = sample.unsqueeze(0)
            knobs_one_hot = torch.cat((knobs_one_hot, sample))
        return np.array(knobs_one_hot)

def get_knob2vec(data, table):
    k2vec = np.zeros((data.shape[0], 22, table.shape[1]))
    for i in range(data.shape[0]):
#         idx = (data[i]==1).nonzero().squeeze().cpu().detach().numpy()
        idx = (data[i]==1).nonzero().squeeze().cpu().detach().numpy()
        k2vec[i] = table[idx]
    return k2vec

def draw_attention(config, target, similar_wk, model_path, save_path=None):
    hidden_size = 128
    batch_size = 32
    KNOB_PATH = '../data/rocksdb_conf'
    EXTERNAL_PATH = '../data/external'
    INTERNAL_PATH = '../data/internal'
    WK_NUM = 16
    compression_type = {'snappy':0, 'zlib':1, 'lz4':2, 'none':3}
    splt_config = config.split(' ')
    column = []
    data = []

    for cfg in splt_config:
        c, v = cfg.split('=')
        column.append(c[1:])
        if c == '-compression_type':
            data.append(compression_type[v])
        else:
            data.append(int(v))
                   
    config = pd.DataFrame(data=data, index=column).T
    
    raw_knobs = rocksdb_knobs_make_dict(KNOB_PATH)
    raw_knobs = pd.DataFrame(data=raw_knobs['data'].astype(np.float32), columns=raw_knobs['columnlabels'])

    internal_dict = {}
    external_dict = {}

    pruned_im = pd.read_csv(os.path.join(INTERNAL_PATH, 'internal_ensemble_pruned_tmp.csv'), index_col=0)
    for wk in range(WK_NUM):
        im = pd.read_csv(os.path.join(INTERNAL_PATH, f'internal_results_{wk}.csv'), index_col=0)
        internal_dict[wk] = im[pruned_im.columns]
    if target > 15:
        im = pd.read_csv(f'../data/target_workload/{target}/internal_results_11.csv', index_col=0)
        internal_dict[wk+1] = im[pruned_im.columns]

    for wk in range(WK_NUM):
        ex = pd.read_csv(os.path.join(EXTERNAL_PATH, f'external_results_{wk}.csv'), index_col=0)
        external_dict[wk] = ex
    if target > 15:
        ex = pd.read_csv(f'../data/target_workload/{target}/external_results_11.csv', index_col=0)
        external_dict[wk+1] = ex
        
    em_tr, em_te = train_test_split(external_dict[similar_wk], test_size=0.2, random_state=22)
    scaler_em = StandardScaler().fit(em_tr)
    
    index_value = get_index_value(raw_knobs)
    conf_one_hot = make_knobsOneHot(index_value, config, raw_knobs)
    table = np.load(f'../data/lookuptable/{similar_wk}/LookupTable.npy')
    conf_knob2vec = get_knob2vec(torch.Tensor(conf_one_hot), table)
    conf_knob2vec = torch.Tensor(conf_knob2vec).cuda()
    
    model = torch.load(f'../model_save/{model_path}')
    model.tf = False
    model.batch_size = batch_size
    
    output, attn_w_ = model(conf_knob2vec.repeat(32, 1, 1))
    
    attn_w = attn_w_.cpu().detach().numpy()
    attn_w = np.average(attn_w, axis=0)
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    im = ax.matshow(attn_w, interpolation='none')
    fig.colorbar(im)

    fontdict = {'fontsize': 14}

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(external_dict[0].columns, fontdict=fontdict, rotation=90)
    ax.set_yticks(np.arange(22))
    ax.set_yticklabels(raw_knobs.columns, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()

    
    return model
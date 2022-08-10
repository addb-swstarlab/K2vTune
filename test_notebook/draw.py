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

def draw_attention(config, similar_wk, model_path, color, title, save_path=None):
    batch_size = 32
    KNOB_PATH = '../data/rocksdb_conf'
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
    
    external_columns = ['TIME', 'RATE', 'WAF', 'SA']
    # index_value = get_index_value(raw_knobs)
    # # conf_one_hot = make_knobsOneHot(index_value, config, raw_knobs)
    conf_one_hot = np.load("../data/knobsOneHot.npy")
    table = np.load(f'../data/lookuptable/{similar_wk}/20000_LookupTable.npy')
    conf_knob2vec = get_knob2vec(torch.Tensor(conf_one_hot), table)
    conf_knob2vec = torch.Tensor(conf_knob2vec).cuda()
    _, k2v_te = train_test_split(conf_knob2vec, test_size=0.2, random_state=22)
    idx = np.random.randint(0, k2v_te.shape[0], batch_size)
    k2v_te = k2v_te[idx]
    model = torch.load(f'../model_save/{model_path}')
    model.tf = False
    model.batch_size = batch_size

    # output, attn_w_ = model(conf_knob2vec.repeat(32, 1, 1))
    output, attn_w = model(k2v_te)
    
    attn_w = attn_w.cpu().detach().numpy()
    attn_w = np.average(attn_w, axis=0)
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    im = ax.matshow(attn_w, interpolation='none', cmap=color)
    fig.colorbar(im)

    fontdict = {'fontsize': 14}

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(external_columns, fontdict=fontdict, rotation=90)
    ax.set_yticks(np.arange(22))
    ax.set_yticklabels(raw_knobs.columns, fontdict=fontdict)
    fontdict = {'fontname': 'Times New Roman', 'fontsize':30, 'fontweight':'bold'}
    ax.set_title(title, fontdict=fontdict, y=-0.075)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    
    plt.show()

    
    return model

def get_max_data(random_data):
    max_idx = []
    for i in range(100):
        idx = random_data['SCORE'][:i+1].argmax()
        max_idx.append(idx)
    return random_data.iloc[max_idx]

def get_default(path):
    def_ex = pd.read_csv(path, index_col=0)
    def_ex = def_ex.T
    return def_ex.to_dict()

def get_score(df, wk):
    def_ex = get_default('data/default_external_metrics.csv')
    d = def_ex[wk]
#     if wk == 16:
#         d = {'TIME':13, 'RATE':23.67, 'WAF':7.9, 'SA':148.21}
#     elif wk == 17:
#         d = {'TIME':43, 'RATE':16.71, 'WAF':11.1, 'SA':288.2}
#     elif wk == 18:
#         d = {'TIME':69.3, 'RATE':14.8, 'WAF':12.4, 'SA':361.19}
    
    df['SCORE'] = (d['TIME']-df['TIME'])/d['TIME'] + (df['RATE']-d['RATE'])/d['RATE'] + \
(d['WAF']-df['WAF'])/d['WAF'] + (d['SA']-df['SA'])/d['SA']
    return df

def draw_plot(x, models, wk, s_size):
#     for m in models['Single'].columns:
    for m in ['SCORE']:
        plt.figure(figsize=(12, 4))
        plt.plot(x, models['Random'][m], marker='_', color='y', label='Random')
#         plt.plot(x, deft[m], marker='_', color='y', label='default')
#         plt.plot(x, fb[m], marker='_', color='y', label='facebook')
#         plt.plot(x, dba[m], marker='_', color='y', label='DBA')
        plt.plot(x, models['Single'][m], marker='|', color='grey', label='Single')
        plt.plot(x, models['Single+Knov2vec'][m], marker='x', color='lightcoral', label='Single+K2v')
        plt.plot(x, models['GRU'][m], marker='*', color='goldenrod', label='GRU')
        plt.plot(x, models['GRU+Attn'][m], marker='.', color='mediumpurple', label='GRU+Attn')
        plt.legend(mode="expand", ncol=8, loc="upper center", bbox_to_anchor=(0, 0.95, 1, 0.2))
#         plt.title(f"workload: {wk} sample size: {s_size} {m}", loc='center')
        fontdict = {'fontname': 'Times New Roman', 'fontsize':30, 'fontweight':'bold'}
        plt.xlabel("Optimization steps", fontdict=fontdict)
        plt.ylabel(m)

def draw_bar(models, wk, s_size):
    metrics = models['DBA'].columns[1:]
    metrics = [metrics[:2], metrics[2:-1]]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for i, metric in enumerate(metrics):
#     for metric in ['SCORE']:
        for j, m in enumerate(metric):
            values = []
            for key in models.keys():
                values.append(models[key].iloc[-1][m])
#             ax[i][j].set_figure((8, 4))
            ax[i][j].grid(axis='y', linestyle='--', which='major', zorder=0)
            ax[i][j].bar(range(len(values)), values, width=0.6, color='lightsteelblue', zorder=2, edgecolor='black', hatch='--')
            ax[i][j].set_xticklabels(models.keys())
            ax[i][j].set_ylabel(m)
#             ax[i][j].set_title(f"workload: {wk} {m}", loc='center')
        
# def wk_draw(wk, s_size, path, plot=False, bar=False):
# #     types = ['Default', 'Facebook', 'DBA', 'RANDOM', 'SingleNet', 'SingleNet+K2v', 'GRU', 'ATTN']
#     models = {}
    
#     data = pd.read_csv(f'{path}{wk}_{s_size}_step.csv', index_col=0)
#     if 'date' in data.columns:
#         data = data.drop(columns=['date'])
    
#     random_data = pd.read_csv(f'data/{wk}_random_step.csv')
#     random_data = random_data.drop(columns=['date'])
    
#     def_data = pd.read_csv(f'data/{wk}_default_step.csv').reset_index(drop=True)
#     fb_data = pd.read_csv(f'data/{wk}_facebook_step.csv').reset_index(drop=True)
#     dba_data = pd.read_csv(f'data/{wk}_dba_step.csv').reset_index(drop=True)
    
#     x = range(100)
#     raw = data.iloc[:100].reset_index(drop=True)
#     dnn = data.iloc[100:200].reset_index(drop=True)
#     gru = data.iloc[200:300].reset_index(drop=True)
#     attn = data.iloc[300:].reset_index(drop=True)
    
#     models['Default'] = get_score(def_data, wk)
#     rand = get_score(random_data, wk)
#     models['Random'] = get_max_data(random_data=rand)
#     models['Facebook'] = get_score(fb_data, wk)
#     models['DBA'] = get_score(dba_data, wk)    
#     models['Single'] = get_score(raw, wk)
#     models['Single+K2v'] = get_score(dnn, wk)
#     models['GRU'] = get_score(gru, wk)
#     models['GRU+Attn'] = get_score(attn, wk)

#     if plot:
#         draw_plot(x, models, wk, s_size)
#     if bar:
#         draw_bar(models, wk, s_size)
        
def get_data(wk, path):
#     types = ['Default', 'Facebook', 'DBA', 'RANDOM', 'SingleNet', 'SingleNet+K2v', 'GRU', 'ATTN']
    models = {}
    
    data = pd.read_csv(f'{path}{wk}_step.csv', index_col=0)
    if 'date' in data.columns:
        data = data.drop(columns=['date'])
    
    random_data = pd.read_csv(f'data/{wk}_random_step.csv')
    random_data = random_data.drop(columns=['date'])
    
    def_data = pd.read_csv(f'data/{wk}_default_step.csv').reset_index(drop=True)
    fb_data = pd.read_csv(f'data/{wk}_facebook_step.csv').reset_index(drop=True)
    dba_data = pd.read_csv(f'data/{wk}_dba_step.csv').reset_index(drop=True)
    cdb_data = get_default('data/cdbtune_res.csv')
    otter_data = get_default('data/ottertune_res.csv')
    
    x = range(100)
    raw = data.iloc[:100].reset_index(drop=True)
    dnn = data.iloc[100:200].reset_index(drop=True)
    gru = data.iloc[200:300].reset_index(drop=True)
    bigru = data.iloc[300:400].reset_index(drop=True)
    attn = data.iloc[400:500].reset_index(drop=True)
    biattn = data.iloc[500:].reset_index(drop=True)
    
    models['Default'] = get_score(def_data, wk)
    rand = get_score(random_data, wk)
    models['Random'] = rand # get_max_data(random_data=rand)
    models['Facebook'] = get_score(fb_data, wk)
    models['DBA'] = get_score(dba_data, wk) 
    models['CDBTune'] = get_score(cdb_data[wk], wk)
    models['OtterTune'] = get_score(otter_data[wk], wk)
    models['Single'] = get_score(raw, wk)
    models['Single+Knov2vec'] = get_score(dnn, wk)
    models['GRU'] = get_score(gru, wk)
    models['BiGRU'] = get_score(bigru, wk)
    models['GRU+Attn'] = get_score(attn, wk)
    models['BiGRU+Attn'] = get_score(biattn, wk)
    
    return models
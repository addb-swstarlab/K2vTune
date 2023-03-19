import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from models.network import RocksDBDataset, SingleNet, GRUNet, EncoderRNN, DecoderRNN, Attention, AttnDecoderRNN
from models.train import train, valid
from models.utils import get_filename
import models.rocksdb_option as option
from scipy.stats import gmean
from sklearn.ensemble import RandomForestRegressor

def euclidean_distance(a, b):
    res = a - b
    res = res ** 2
    res = np.sqrt(res)
    return np.average(res)

def get_euclidean_distance(internal_dict, logger, opt):
    scaler = MinMaxScaler().fit(pd.concat(internal_dict))
    target_config_idx = [1004,1214,1312,2465,3306,3333,4569,4801,5124,5389,8490,9131,9143,11896,12065,12293,12491,13098,18088,19052]
    
    trg = opt.target
    if trg > 15:
        trg = 16
    
    wk = []
    # for im_d in internal_dict:
    #     wk.append(scaler.transform(internal_dict[im_d].iloc[:opt.target_size, :]))
    for im_d in internal_dict:
        if im_d == 16: # target workload
            wk.append(scaler.transform(internal_dict[im_d]))
        else:
            wk.append(scaler.transform(internal_dict[im_d].iloc[target_config_idx, :]))

    big = 100
    for i in range(len(wk)):
        ed = euclidean_distance(wk[trg], wk[i])
        if ed<big and trg != i: 
            big=ed
            idx = i
        logger.info(f'{i:4}th   {ed:.5f}')
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
    
    np.save(os.path.join(knobs.TABLE_PATH, str(knobs.s_wk), f'{opt.sample_size}_LookupTable.npy'), table)
    
    return table

def train_fitness_function(knobs, logger, opt):
    if opt.mode == 'RF':
        rf = RandomForestRegressor(max_depth=2, random_state=0)
        rf.fit(knobs.norm_k_tr.cpu().detach().numpy(), knobs.norm_em_tr.cpu().detach().numpy())
        return rf, rf.predict(knobs.norm_k_te.cpu().detach().numpy())

    if opt.mode == 'dnn':
        Dataset_K2vec_tr = RocksDBDataset(torch.reshape(knobs.knob2vec_tr, (knobs.knob2vec_tr.shape[0], -1)), knobs.norm_em_tr)
        Dataset_K2vec_te = RocksDBDataset(torch.reshape(knobs.knob2vec_te, (knobs.knob2vec_te.shape[0], -1)), knobs.norm_em_te)
    elif opt.mode == 'raw' or opt.mode == 'RF':
        Dataset_K2vec_tr = RocksDBDataset(knobs.norm_k_tr, knobs.norm_em_tr)
        Dataset_K2vec_te = RocksDBDataset(knobs.norm_k_te, knobs.norm_em_te)
    else: # gru, attngru
        Dataset_K2vec_tr = RocksDBDataset(knobs.knob2vec_tr, knobs.norm_em_tr)
        Dataset_K2vec_te = RocksDBDataset(knobs.knob2vec_te, knobs.norm_em_te)

    loader_K2vec_tr = DataLoader(dataset = Dataset_K2vec_tr, batch_size = opt.batch_size, shuffle=True)
    loader_K2vec_te = DataLoader(dataset = Dataset_K2vec_te, batch_size = opt.batch_size, shuffle=False)

    if opt.mode == 'dnn':
        model = SingleNet(input_dim=torch.reshape(knobs.knob2vec_tr, (knobs.knob2vec_tr.shape[0], -1)).shape[-1], hidden_dim=opt.hidden_size, output_dim=knobs.norm_em_tr.shape[-1]).cuda()
    elif opt.mode == 'gru':
        encoder = EncoderRNN(input_dim=knobs.knob2vec_tr.shape[-1], hidden_dim=opt.hidden_size, bidirect=opt.bidirect)
        decoder = DecoderRNN(input_dim=1, hidden_dim=opt.hidden_size, output_dim=1, bidirect=opt.bidirect)
        model = GRUNet(encoder=encoder, decoder=decoder, tf=opt.tf, batch_size=opt.batch_size).cuda()
    elif opt.mode == 'attngru':
        attn = Attention(batch_size=opt.batch_size, hidden_size=opt.hidden_size, method=opt.attn_mode, mlp=False)
        encoder = EncoderRNN(input_dim=knobs.knob2vec_tr.shape[-1], hidden_dim=opt.hidden_size, bidirect=opt.bidirect)
        decoder = AttnDecoderRNN(input_dim=1, hidden_dim=opt.hidden_size, output_dim=1, attn=attn, bidirect=opt.bidirect)
        model = GRUNet(encoder=encoder, decoder=decoder, tf=opt.tf, batch_size=opt.batch_size).cuda()
    elif opt.mode == 'raw':
        model = SingleNet(input_dim=knobs.norm_k_tr.shape[1], hidden_dim=16, output_dim=knobs.norm_em_tr.shape[-1]).cuda()

    if opt.train:       
        logger.info(f"[Train MODE] Training Model") 
        best_loss = 100
        name = get_filename('model_save', 'model', '.pt')
        for epoch in range(opt.epochs):
            loss_tr = train(model, loader_K2vec_tr, opt.lr)
            if opt.mode != 'dnn': model.tf = False
            loss_te, outputs = valid(model, loader_K2vec_te)
            if opt.mode != 'dnn': model.tf = opt.tf
        
            logger.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")

            if best_loss > loss_te and epoch>15:
                best_loss = loss_te
                best_model = model
                best_outputs = outputs
                torch.save(best_model, os.path.join('model_save', name))
        logger.info(f"loss is {best_loss:.4f}, save model to {os.path.join('model_save', name)}")
        

        return best_model, best_outputs
    elif opt.eval:
        logger.info(f"[Eval MODE] Trained Model Loading with path: {opt.model_path}")
        model = torch.load(os.path.join('model_save',opt.model_path))
        _, outputs = valid(model, loader_K2vec_te)
        return model, outputs

def score_function(df, pr, ex_w):
    if df.size > 1:
        df = df.squeeze()
    score = (df[0] - pr[0]) * ex_w[0] + (pr[1] - df[1]) * ex_w[1] + (df[2] - pr[2]) * ex_w[2] + (df[3] - pr[3]) * ex_w[3]
    return round(score, 6)

def set_fitness_function(solution, model, knobs, opt):
    Dataset_sol = RocksDBDataset(solution, np.zeros((len(solution), 1)))
    loader_sol = DataLoader(Dataset_sol, shuffle=False, batch_size=opt.GA_batch_size)

    
    ## Set phase
    model.eval()
    model.batch_size = opt.GA_batch_size
    model.tf = False # Teacher Forcing Off
    
    ## Predict
    fitness_f = []
    with torch.no_grad():
        for data, _ in loader_sol:
            if opt.mode == 'dnn':
                data = torch.reshape(data, (data.shape[0], -1))
            fitness_batch, _ = model(data)
            # fitness_batch = knobs.scaler_em.inverse_transform(fitness_batch.cpu().numpy()) # if fitness_batch's type is torch.Tensor()
            fitness_batch = fitness_batch.cpu().numpy()
            fitness_batch = [score_function(knobs.default_trg_em, _, opt.ex_weight) for _ in fitness_batch]
            fitness_f += fitness_batch # [1,2] += [3,4,5] --> [1,2,3,4,5]
    
    return fitness_f

def GA_optimization(knobs, fitness_function, logger, opt):    
    configs = knobs.knobs#.to_numpy()
    n_configs = configs.shape[1]
    n_pool_half = int(opt.pool/2)
    mutation = int(n_configs * 0.4)

    ## use pool nums of top to initialize current_solution_pool
    scaled_em = knobs.scaler_em.transform(knobs.s_external_metrics)
    em_score = [score_function(knobs.default_trg_em, _, opt.ex_weight) for _ in scaled_em]
    em_score = [_*-1 for _ in em_score]
    idx_em = np.argsort(em_score)[:opt.pool]
    
    current_solution_pool = pd.DataFrame(configs.to_numpy()[idx_em, :], columns=knobs.columns)
    # current_solution_pool = configs[:opt.pool] # dataframe
    # current_solution_pool = torch.Tensor(current_solution_pool.to_numpy()).cuda() # if current_solution_pool's type is dataframe
    step_best_solution = []
    step_best_fitness = []
    
    for i in range(opt.generation):
        if opt.mode == 'raw':
            scaled_pool = torch.Tensor(knobs.scaler_k.transform(current_solution_pool)).cuda()
            fitness = set_fitness_function(scaled_pool, fitness_function, knobs, opt)
        elif opt.mode == 'RF':
            scaled_pool = knobs.scaler_k.transform(current_solution_pool)
            fitness = fitness_function.predict(scaled_pool)
            fitness = [score_function(knobs.default_trg_em, _, opt.ex_weight) for _ in fitness]
        else:
            onehot_pool = knobs.load_knobsOneHot(k=current_solution_pool, save=False)
            onehot_pool = torch.Tensor(onehot_pool).cuda()
            k2v_pool = knobs.get_knob2vec(onehot_pool, knobs.lookuptable)
            k2v_pool = torch.Tensor(k2v_pool).cuda()
            fitness = set_fitness_function(k2v_pool, fitness_function, knobs, opt)

        fitness = [_*-1 for _ in fitness] # bigger is good

        ## best parents selection
        idx_fitness = np.argsort(fitness)[:n_pool_half]
        best_solution_pool = current_solution_pool.to_numpy()[idx_fitness, :] # dataframe -> numpy
        if i % 10 == 9:
            logger.info(f"[{i+1:3d}/{opt.generation:3d}] best fitness: {fitness[idx_fitness[0]]:.5f}")
        
        ## cross-over and mutation
        new_solution_pool = np.zeros_like(best_solution_pool) # new_solution_pool.shape = (n_pool_half,22)
        for j in range(n_pool_half):
            pivot = np.random.choice(np.arange(1,n_configs))
            # pivot = np.random.choice(np.arange(1,n_configs-1))
            new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
            new_solution_pool[j][pivot:] = best_solution_pool[n_pool_half-1-j][pivot:]
            
            _, random_knobs = option.make_random_option()

            random_knobs = {_:random_knobs[_] for _ in knobs.columns}
            random_list_knobs = list(random_knobs.values())
            random_knob_index = np.arange(n_configs)
            np.random.shuffle(random_knob_index)
            random_knob_index = random_knob_index[:mutation]
            # random_knob_index = [random.randint(0,21) for r in range(mutation)]
            for k in range(len(random_knob_index)):
                new_solution_pool[j][random_knob_index[k]] = random_list_knobs[random_knob_index[k]]
        
        ## stack
        current_solution_pool = np.vstack([best_solution_pool, new_solution_pool]) # current_solution_pool.shape = (n_pool,22)
        current_solution_pool = pd.DataFrame(current_solution_pool, columns=knobs.columns)
        
        if (i+1)%25==0:
            SOLUTION_PATH = '/home/jieun/Knob2Vec/save_solution'
            if not os.path.isdir(SOLUTION_PATH):
                os.makedirs(SOLUTION_PATH)
            if not os.path.isdir(os.path.join(SOLUTION_PATH, datetime.today().strftime("%Y%m%d"))):
                os.makedirs(os.path.join(SOLUTION_PATH, datetime.today().strftime("%Y%m%d")))
            current_solution_pool.to_csv(os.path.join(SOLUTION_PATH, datetime.today().strftime("%Y%m%d"), f'gen{i+1}_{opt.target}_{opt.mode}_{opt.bidirect}.csv'))
        
        ## save best solution at each step
        step_best_solution.append(best_solution_pool[0])
        step_best_fitness.append(-np.min(fitness))
        
    final_solution = best_solution_pool[0]
    recommend_command = ''
    step_recommend_command = []
    
    for idx, col in enumerate(knobs.columns):                 
        recommend_command = convert_int_to_category(col, recommend_command, final_solution[idx])
        
    recommend_command = make_dbbench_command(opt.target, recommend_command)

    logger.info(f"db_bench command is  {recommend_command}")
    
    for step in range(len(step_best_solution)):
        cmd = ''
        step_solution = step_best_solution[step]
        for idx, col in enumerate(knobs.columns):
            cmd = convert_int_to_category(col, cmd, step_solution[idx])
        cmd = make_dbbench_command(opt.target, cmd)
        step_recommend_command.append(cmd)
    
    return recommend_command, step_recommend_command, step_best_fitness

def convert_int_to_category(col, cmd, s):
    if col=='compression_type':
            ct = ["snappy", "zlib", "lz4", "none"]
            cmd += f'-{col}={ct[int(s)]} '
    else:
        cmd += f'-{col}={int(s)} '
    return cmd

def make_dbbench_command(trg_wk, rc_cmd):
    wk_info = pd.read_csv('data/rocksdb_workload_info.csv', index_col=0)
    f_wk_info = wk_info.loc[:,:'num']
    b_wk_info = wk_info.loc[:, 'benchmarks':]
    cmd = 'rocksdb/db_bench '   
    f_cmd = " ".join([f'-{_}={int(f_wk_info.loc[trg_wk][_])}' for _ in f_wk_info.columns])
    b_cmd = f"--{b_wk_info.columns[0]}={b_wk_info.loc[trg_wk][0]} "
    b_cmd += f"--statistics "
    if not np.isnan(b_wk_info.loc[trg_wk][1]):
        b_cmd += f"--{b_wk_info.columns[1]}={int(b_wk_info.loc[trg_wk][1])}"

    cmd += f_cmd + " " + rc_cmd + b_cmd + ' > /jieun/result.txt'

    return cmd

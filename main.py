import os
import argparse
import pandas as pd
import numpy as np
from models.utils import get_logger, rocksdb_knobs_make_dict
import models.steps
from models.knobs import Knob
from models.steps import get_euclidean_distance, train_knob2vec, train_fitness_function
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from benchmark import exec_benchmark
from datetime import datetime
from top_knobs import TOP_5K_COLUMNS, TOP_10K_COLUMNS

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=1, help='Choose target workload')
parser.add_argument('--target_size', type=int, default=10, help='Define target workload size')
parser.add_argument('--lr', type=float, default=0.001, help='Define learning rate')
parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')
parser.add_argument('--hidden_size', type=int, default=64, help='Define model hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='Define model batch size')
parser.add_argument('--mode', type=str, default='gru', help='choose which model be used on fitness function')
parser.add_argument('--attn_mode', type=str, choices=['dot', 'general', 'concat', 'bahdanau'], default='general', help='choose which attention be used')
parser.add_argument('--tf', action='store_true', help='Choose usage of teacher forcing. if trigger this, tf be true')
parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')
parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
parser.add_argument('--model_path', type=str, help='Define which .pt will be loaded on model')
parser.add_argument('--pool', type=int, default=128, help='Define the number of pool to GA algorithm')
parser.add_argument('--generation', type=int, default=1000, help='Define the number of generation to GA algorithm')
parser.add_argument('--GA_batch_size', type=int, default=32, help='Define GA batch size')
parser.add_argument('--save', action='store_true', help='Choose save the score on csv file or just show')
parser.add_argument('--step', action='store_true', help='If want to see stepped results, trigger this')
parser.add_argument('--sample_size', type=int, default=20000, help='Define train sample size, max is 20000')
parser.add_argument('--bidirect', action='store_true', help='Choose whether applying bidirectional GRU')
parser.add_argument('--optimization', type=str, default='ga', choices=['ga', 'smac'], help='Define which .pt will be loaded on model')
parser.add_argument('--wm_mode', type=str, default='internal', choices=['corr', 'internal', 'external'], help='Define the mode to calculate workload similarities')
parser.add_argument('--similar_wk', type=int, default=None, help='If this parameter is defined, workload mapping is skipped')
parser.add_argument('--topK', type=int, default=None, help='Tuning only top-k knobs')

opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists('model_save'):
    os.mkdir('model_save')

import logging
logger = logging.getLogger()

# logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')

KNOB_PATH = 'data/rocksdb_conf'
EXTERNAL_PATH = 'data/external'
INTERNAL_PATH = 'data/internal'
WK_NUM = 16

def main():
    logger.info("## get raw datas ##")
    raw_knobs = rocksdb_knobs_make_dict(KNOB_PATH)
    raw_knobs = pd.DataFrame(data=raw_knobs['data'].astype(np.float32), columns=raw_knobs['columnlabels'])  
    
    if opt.topK == 5:
        raw_knobs = raw_knobs[TOP_5K_COLUMNS[opt.target]]
    elif opt.topK == 10:
        raw_knobs = raw_knobs[TOP_10K_COLUMNS[opt.target]]
    
    internal_dict = {}
    external_dict = {}

    pruned_im = pd.read_csv(os.path.join(INTERNAL_PATH, 'internal_ensemble_pruned_tmp.csv'), index_col=0)
    for wk in range(WK_NUM):
        im = pd.read_csv(os.path.join(INTERNAL_PATH, f'internal_results_{wk}.csv'), index_col=0)
        internal_dict[wk] = im[pruned_im.columns]
    if opt.target > 15:
        im = pd.read_csv(f'data/target_workload/{opt.target}/internal_results_{opt.target}.csv', index_col=0)
        internal_dict[wk+1] = im[pruned_im.columns]

    for wk in range(WK_NUM):
        ex = pd.read_csv(os.path.join(EXTERNAL_PATH, f'external_results_{wk}.csv'), index_col=0)
        external_dict[wk] = ex
    if opt.target > 15:
        ex = pd.read_csv(f'data/target_workload/{opt.target}/external_results_{opt.target}.csv', index_col=0)
        external_dict[wk+1] = ex
    logger.info('## get raw datas DONE ##')


    knobs = Knob(raw_knobs, internal_dict, external_dict, opt, opt.target, opt.sample_size)

    logger.info("## Workload Mapping ##")
    if opt.similar_wk is not None:
        logger.info("$$ Skip Workload Mapping $$")
        similar_wk = opt.similar_wk
    else:
        similar_wk = get_euclidean_distance(internal_dict, external_dict, logger, opt)
    logger.info("## Workload Mapping DONE##")


    logger.info("## Configuration Recommendation ##")
    knobs.split_data(similar_wk)
    knobs.scale_data()
    logger.info("## Train Knob2Vec for similar workload ##")
    # if there is pre-trained model results, just load numpy file or train model and get table results
    if opt.topK is not None:
        LOOKUPTABLE_PATH = os.path.join('data/lookuptable', str(knobs.s_wk), f'{opt.target}_{opt.sample_size}_top{opt.topK}_LookupTable.npy')
    else:
        LOOKUPTABLE_PATH = os.path.join('data/lookuptable', str(knobs.s_wk), f'{opt.target}_{opt.sample_size}_LookupTable.npy')
    if os.path.exists(LOOKUPTABLE_PATH):
        logger.info("lookup table file is already existed. Load the file")
        knobs.set_lookuptable(np.load(LOOKUPTABLE_PATH))
    else:
        logger.info("lookup table file is not existed. Train knob2vec model")
        knobs.set_lookuptable(train_knob2vec(knobs=knobs, logger=logger, opt=opt, path=LOOKUPTABLE_PATH))
    knobs.set_knob2vec()
    logger.info("## Train Knob2Vec for similar workload DONE##")
  
    if opt.train:
        logger.info("## Train Fitness Function ##")
        fitness_function, outputs = train_fitness_function(knobs=knobs, logger=logger, opt=opt)

        # if outputs' type are torch.tensor
        pred = np.round(knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
        # if outputs' type are numpy array
        # pred = np.round(knobs.scaler_em.inverse_transform(outputs), 2)
        true = knobs.em_te.to_numpy()

        for i in range(10):
            logger.info(f'predict rslt: {pred[i]}')
            logger.info(f'ground truth: {true[i]}\n')
    
    elif opt.eval:
        logger.info("## Load Trained Fitness Function ##")
        fitness_function, outputs = train_fitness_function(knobs=knobs, logger=logger, opt=opt)
        pred = np.round(knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
        true = knobs.em_te.to_numpy()
        
    else:
        logger.exception("Choose Model mode, '--train' or '--eval'")      

    r2_res = r2_score(true, pred, multioutput='raw_values')
    logger.info('[R2 SCORE]')
    logger.info(f'TIME:{r2_res[0]:.4f}, RATE:{r2_res[1]:.4f}, WAF:{r2_res[2]:.4f}, SA:{r2_res[3]:.4f}')
    r2_res = np.average(r2_res)
    logger.info(f'average r2 score = {r2_res:.4f}')
    pcc_res = np.zeros(4)
    ci_res = np.zeros(4)
    for idx in range(4):
        res, _ = pearsonr(true[:,idx], pred[:,idx])  
        pcc_res[idx] = res
        res = concordance_index(true[:,idx], pred[:,idx])
        ci_res[idx] = res
    # pcc_res = pcc_res/len(true)
    # ci_res = ci_res/len(true)
    logger.info('[PCC SCORE]')
    logger.info(f'TIME:{pcc_res[0]:.4f}, RATE:{pcc_res[1]:.4f}, WAF:{pcc_res[2]:.4f}, SA:{pcc_res[3]:.4f}')
    logger.info(f'average pcc score = {np.average(pcc_res):.4f}')
    logger.info('[CI SCORE]')
    logger.info(f'TIME:{ci_res[0]:.4f}, RATE:{ci_res[1]:.4f}, WAF:{ci_res[2]:.4f}, SA:{ci_res[3]:.4f}')
    logger.info(f'average ci score = {np.average(ci_res):.4f}')
        

    
    file_name = f"{datetime.today().strftime('%Y%m%d')}_{opt.sample_size}_prediction_score.csv"
    if os.path.isfile(file_name) is False:
        pd.DataFrame(data=['r2', 'pcc', 'ci'], columns=['score']).to_csv(file_name, index=False)
    pred_score = pd.read_csv(file_name, index_col=0)
    if opt.bidirect:
        pred_score[f'{opt.target}_bi{opt.mode}'] = [r2_res, pcc_res, ci_res]
    else:
        pred_score[f'{opt.target}_{opt.mode}'] = [r2_res, pcc_res, ci_res]
    pred_score.to_csv(file_name)
    
    recommend_command, _, _ = getattr(models.steps, f"{opt.optimization.upper()}_optimization")(knobs=knobs, fitness_function=fitness_function, logger=logger, opt=opt)

    logger.info("## Train/Load Fitness Function DONE ##")
    logger.info("## Configuration Recommendation DONE ##")
    
    exec_benchmark(recommend_command, opt)

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()

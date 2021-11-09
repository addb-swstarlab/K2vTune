import os
import argparse
import pandas as pd
import numpy as np
from models.utils import get_logger, rocksdb_knobs_make_dict
from models.knobs import Knob
from models.steps import get_euclidean_distance, train_knob2vec, train_fitness_function, GA_optimization
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from benchmark import exec_benchmark

os.system('clear')

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=1, help='Choose target workload')
parser.add_argument('--target_size', type=int, default=10, help='Define target workload size')
parser.add_argument('--lr', type=float, default=0.001, help='Define learning rate')
parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')
parser.add_argument('--hidden_size', type=int, default=64, help='Define model hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='Define model batch size')
parser.add_argument('--mode', type=str, default='gru', help='choose which model be used on fitness function')
parser.add_argument('--attn_mode', type=str, choices=['dot', 'general', 'concat', 'bahdanau'], default='dot', help='choose which attention be used')
parser.add_argument('--tf', action='store_true', help='Choose usage of teacher forcing. if trigger this, tf be true')
parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')
parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
parser.add_argument('--model_path', type=str, help='Define which .pt will be loaded on model')
parser.add_argument('--pool', type=int, default=128, help='Define the number of pool to GA algorithm')
parser.add_argument('--generation', type=int, default=1000, help='Define the number of generation to GA algorithm')
parser.add_argument('--GA_batch_size', type=int, default=32, help='Define GA batch size')
parser.add_argument('--ex_weight', type=float, action='append', help='Define external metrics weight to calculate score')
parser.add_argument('--save', action='store_true', help='Choose save the score on csv file or just show')

opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')

logger, log_dir = get_logger(os.path.join('./logs'))

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

    internal_dict = {}
    external_dict = {}

    pruned_im = pd.read_csv(os.path.join(INTERNAL_PATH, 'internal_ensemble_pruned_tmp.csv'), index_col=0)
    for wk in range(WK_NUM):
        im = pd.read_csv(os.path.join(INTERNAL_PATH, f'internal_results_{wk}.csv'), index_col=0)
        internal_dict[wk] = im[pruned_im.columns]
    if opt.target > 15:
        im = pd.read_csv(f'data/target_workload/{opt.target}/internal_results_11.csv', index_col=0)
        internal_dict[wk+1] = im[pruned_im.columns]

    for wk in range(WK_NUM):
        ex = pd.read_csv(os.path.join(EXTERNAL_PATH, f'external_results_{wk}.csv'), index_col=0)
        external_dict[wk] = ex
    if opt.target > 15:
        ex = pd.read_csv(f'data/target_workload/{opt.target}/external_results_11.csv', index_col=0)
        external_dict[wk+1] = ex
    logger.info('## get raw datas DONE ##')


    knobs = Knob(raw_knobs, internal_dict, external_dict, opt.target)


    logger.info("## Workload Mapping ##")
    similar_wk = get_euclidean_distance(internal_dict, logger, opt)
    logger.info("## Workload Mapping DONE##")


    logger.info("## Configuration Recommendation ##")
    knobs.split_data(similar_wk)
    knobs.scale_data()
    logger.info("## Train Knob2Vec for similar workload ##")
    # if there is pre-trained model results, just load numpy file or train model and get table results
    LOOKUPTABLE_PATH = os.path.join('data/lookuptable', str(knobs.s_wk), 'LookupTable.npy')
    if os.path.exists(LOOKUPTABLE_PATH):
        logger.info("lookup table file is already existed. Load the file")
        knobs.set_lookuptable(np.load(LOOKUPTABLE_PATH))
    else:
        logger.info("lookup table file is not existed. Train knob2vec model")
        knobs.set_lookuptable(train_knob2vec(knobs=knobs, logger=logger, opt=opt))
    knobs.set_knob2vec()
    logger.info("## Train Knob2Vec for similar workload DONE##")
  
    if opt.train:
        logger.info("## Train Fitness Function ##")
        fitness_function, outputs = train_fitness_function(knobs=knobs, logger=logger, opt=opt)

        # if outputs' type are torch.tensor
        # pred = np.round(knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
        # if outputs' type are numpy array
        pred = np.round(knobs.scaler_em.inverse_transform(outputs), 2)
        true = knobs.em_te.to_numpy()

        for i in range(10):
            logger.info(f'predict rslt: {pred[i]}')
            logger.info(f'ground truth: {true[i]}\n')

        r2_res = r2_score(true, pred, multioutput='raw_values')
        logger.info(f'average r2 score = {np.average(r2_res):.4f}')
        pcc_res = 0
        ci_res = 0
        for idx in range(len(true)):
            res, _ = pearsonr(true[idx], pred[idx])
            pcc_res += res
            res = concordance_index(true[idx], pred[idx])
            ci_res += res
        logger.info(f'average pcc score = {pcc_res/len(true):.4f}')
        logger.info(f'average ci score = {ci_res/len(true):.4f}')
        
    elif opt.eval:
        logger.info("## Load Trained Fitness Function ##")
        fitness_function, outputs = train_fitness_function(knobs=knobs, logger=logger, opt=opt)
        pred = np.round(knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
        true = knobs.em_te.to_numpy()

        r2_res = r2_score(true, pred, multioutput='raw_values')
        logger.info(f'average r2 score = {np.average(r2_res):.4f}')
        pcc_res = 0
        ci_res = 0
        for idx in range(len(true)):
            res, _ = pearsonr(true[idx], pred[idx])
            pcc_res += res
            res = concordance_index(true[idx], pred[idx])
            ci_res += res
        logger.info(f'average pcc score = {pcc_res/len(true):.4f}')
        logger.info(f'average ci score = {ci_res/len(true):.4f}')
        
    else:
        logger.exception("Choose Model mode, '--train' or '--eval'")
    
    recommend_command = GA_optimization(knobs=knobs, fitness_function=fitness_function, logger=logger, opt=opt)

    logger.info("## Train/Load Fitness Function DONE ##")
    
    logger.info("## Configuration Recommendation DONE ##")

    ## Execute db_benchmark with recommended commands by transporting to other server
    exec_benchmark(recommend_command, opt)



if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()

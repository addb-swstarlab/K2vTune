import os
import argparse
import pandas as pd
import numpy as np
from models.utils import get_logger, rocksdb_knobs_make_dict
from models.knobs import Knob
from models.steps import get_euclidean_distance, train_knob2vec, train_fitness

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=0, help='Choose target workload')
parser.add_argument('--target_size', type=int, default=10, help='Define target workload size')

opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')

logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}:\t {vars(opt)[i]}')

KNOB_PATH = '../data/rocksdb_conf'
EXTERNAL_PATH = '../data/external'
INTERNAL_PATH = '../data/internal'
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

    for wk in range(WK_NUM):
        ex = pd.read_csv(os.path.join(EXTERNAL_PATH, f'external_results_{wk}.csv'), index_col=0)
        external_dict[wk] = ex
    logger.info('## get raw datas DONE ##')


    knobs = Knob(raw_knobs, internal_dict, external_dict, opt.target)


    logger.info("## Workload Mapping ##")
    similar_wk = get_euclidean_distance(internal_dict, opt.target, opt.target_size, logger)
    logger.info("## Workload Mapping DONE##")


    logger.info("## Configuration Recommendation ##")
    knobs.split_data(similar_wk)
    knobs.scale_data()
    logger.info("## Train Knob2Vec for similar workload ##")
    # if there is pre-trained model results, just load numpy file or train model and get table results
    LOOKUPTABLE_PATH = os.path.join('../data/lookuptable', str(knobs.s_wk), 'LookupTable.npy')
    if os.path.exists(LOOKUPTABLE_PATH):
        knobs.set_lookuptable(np.load(LOOKUPTABLE_PATH))
    else:
        knobs.set_lookuptable(train_knob2vec(knobs=knobs, logger=logger))
    knobs.set_knob2vec()
    logger.info("## Train Knob2Vec for similar workload DONE##")


    logger.info("## Train Fitness Function ##")
    fitness_function = train_fitness(knobs=knobs, logger=logger)
    logger.info("## Train Fitness Function DONE ##")
    
    logger.info("## Configuration Recommendation DONE ##")

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
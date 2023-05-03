import os, logging
import pandas as pd
from datetime import datetime
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from models.rocksdb_option import def_option, KB, MB
import torch

class Knob2vec_SMAC(object):
    def __init__(self, knobs, predictive_model, opt, config_info_path=None):
        self.opt = opt
        self.knobs = knobs
        self.predictive_model = predictive_model
        self.config_info_path = config_info_path

        self._get_scenario_path_name()
        self._init_scenario()
        
    def _eval_model(self):
        self.predictive_model.eval()
        self.predictive_model.tf = False
    
    def _get_scenario_path_name(self):
        i = 0
        date_str = datetime.today().strftime("%Y%m%d")
        name = date_str + '-' + '%02d'%i
        while os.path.isdir(os.path.join('smac3_output', name)):
            i += 1
            name = date_str + '-' + '%02d'%i
        self.scenario_name = name
        logging.info("#########################################################")
        logging.info("Scenario data will saved in {}".format(os.path.join('smac3_output', self.scenario_name)))
        logging.info("#########################################################\n")        
    
    def _init_scenario(self):
        self.cs = ConfigurationSpace()
        hyps = []
        hyps.append(UniformIntegerHyperparameter("max_background_compactions", lower=1, upper=16, default_value=def_option["max_background_compactions"]))
        hyps.append(UniformIntegerHyperparameter("max_background_flushes", lower=1, upper=16, default_value=def_option["max_background_flushes"]))
        hyps.append(UniformIntegerHyperparameter("write_buffer_size", lower=512*KB, upper=2048*KB, q=KB, default_value=def_option["write_buffer_size"]))
        hyps.append(UniformIntegerHyperparameter("max_write_buffer_number", lower=2, upper=8, default_value=def_option["max_write_buffer_number"]))
        hyps.append(UniformIntegerHyperparameter("min_write_buffer_number_to_merge", lower=1, upper=2, default_value=def_option["min_write_buffer_number_to_merge"]))
        hyps.append(CategoricalHyperparameter("compaction_pri", choices=[0, 1, 2, 3], default_value=def_option["compaction_pri"]))
        hyps.append(CategoricalHyperparameter("compaction_style", choices=[0, 1, 2, 3], default_value=def_option["compaction_style"]))
        hyps.append(UniformIntegerHyperparameter("level0_file_num_compaction_trigger", lower=2, upper=8, default_value=def_option["level0_file_num_compaction_trigger"]))
        hyps.append(UniformIntegerHyperparameter("level0_slowdown_writes_trigger", lower=16, upper=32, default_value=def_option["level0_slowdown_writes_trigger"]))
        hyps.append(UniformIntegerHyperparameter("level0_stop_writes_trigger", lower=32, upper=64, default_value=def_option["level0_stop_writes_trigger"]))
        hyps.append(CategoricalHyperparameter("compression_type", choices=[0, 1, 2, 3], default_value=def_option["compression_type"]))
        hyps.append(UniformIntegerHyperparameter("bloom_locality", lower=0, upper=1, default_value=def_option["bloom_locality"]))
        hyps.append(CategoricalHyperparameter("open_files", choices=[-1, 10000, 100000, 1000000], default_value=def_option["open_files"]))
        hyps.append(UniformIntegerHyperparameter("block_size", lower=2*KB, upper=16*KB, q=KB, default_value=def_option["block_size"]))
        hyps.append(UniformIntegerHyperparameter("cache_index_and_filter_blocks", lower=0, upper=1, default_value=def_option["cache_index_and_filter_blocks"]))
        hyps.append(UniformIntegerHyperparameter("max_bytes_for_level_base", lower=2*MB, upper=8*MB, q=MB, default_value=def_option["max_bytes_for_level_base"]))
        hyps.append(UniformIntegerHyperparameter("max_bytes_for_level_multiplier", lower=8, upper=12, default_value=def_option["max_bytes_for_level_multiplier"]))
        hyps.append(UniformIntegerHyperparameter("target_file_size_base", lower=512*KB, upper=2048*KB, q=KB, default_value=def_option["target_file_size_base"]))
        hyps.append(UniformIntegerHyperparameter("target_file_size_multiplier", lower=1, upper=2, default_value=def_option["target_file_size_multiplier"]))
        hyps.append(UniformIntegerHyperparameter("num_levels", lower=5, upper=8, default_value=def_option["num_levels"]))
        hyps.append(UniformFloatHyperparameter("memtable_bloom_size_ratio", lower=0, upper=0.2, q=0.05, default_value=def_option["memtable_bloom_size_ratio"]))
        hyps.append(UniformFloatHyperparameter("compression_ratio", lower=0, upper=0.99, q=0.01, default_value=def_option["compression_ratio"]))
        
        self.cs.add_hyperparameters(hyps)
        self.scenario = Scenario(self.cs, deterministic=True, n_trials=self.opt.generation, name=self.scenario_name)
        
        self._target_function(self.cs.sample_configuration())
        
    def _target_function(self, config: Configuration, seed: int = 0) -> float:
        config['compaction_style'] = 0
        X = pd.DataFrame.from_dict(config.get_dictionary(), orient='index').T        
        
        onehot_X = self.knobs.load_knobsOneHot(k=X, save=False)
        onehot_X = torch.Tensor(onehot_X).cuda()
        k2v_X = self.knobs.get_knob2vec(onehot_X, self.knobs.lookuptable)
        k2v_X = torch.Tensor(k2v_X).cuda() # (1, 1, 128)
        print(k2v_X.shape)
        k2v_X = k2v_X.repeat(1, self.opt.batch_size, 1) # fit size to (1, batch, 128)
        print(k2v_X.shape)
        with torch.no_grad():
            res = self.predictive_model(k2v_X)
        
        print(res)
        
        assert False
        
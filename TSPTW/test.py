
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
import logging
import numpy as np
from DeCo.utils.utils import create_logger, copy_all_src
from BTWTester import TSPTester as Tester

#############################################################

# testing problem size
problem_size = 50

model_load_path='pretrained/DeCo50' 
model_load_epoch=100
# model_load_path='pretrained/DeCo100'#Deco_100
# model_load_epoch=200

mode = 'test'

##########################################################################################

b = os.path.abspath(".").replace('\\', '/')
test_batch=1000
env_params = {
    'mode': mode,
    # 其他规模TSPTW
    # TSPTW50
    # 'data_path':f"./data/tsptw50_easy.pkl",
    # "solution_path": f"./data/lkh_tsptw50_easy.pkl",
    # 'data_path':f"./data/tsptw50_medium.pkl",
    # "solution_path": f"./data/lkh_tsptw50_medium.pkl",
    'data_path':f"./data/tsptw50_hard.pkl",
    "solution_path": f"./data/lkh_tsptw50_hard.pkl", 

    # # TSPTW 100
    # 'data_path':f"./data/tsptw100_easy.pkl",
    # "solution_path": f"./data/lkh_tsptw100_easy.pkl", 
    # 'data_path':f"./data/tsptw100_medium.pkl",
    # "solution_path": f"./data/lkh_tsptw100_medium.pkl", 
    # 'data_path':f"./data/tsptw100_hard.pkl",
    # "solution_path": f"./data/lkh_tsptw100_hard.pkl", 
    
    'sub_path': False,
    'batch_size':test_batch
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

tester_params = {
    'tw_norm':False,
    'eval_type':'greedy',
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': 10000,
    'test_batch_size': test_batch, 
    'Sample': False,
    'sample_size':8,  #>8    
    # example: 32 means aug:8 samplesize:4; 8 means aug:8 samplesize:1
    'augment': True, #sample 
    'augment_batch_size': test_batch,
}

sample_batch_size=test_batch
if tester_params['augment']:
    tester_params['augment_batch_size']=sample_batch_size
    env_params['batch_size']=sample_batch_size
    tester_params['test_batch_size']=sample_batch_size
logger_params = {
    'log_file': {
        'desc': f'test__tsp{problem_size}',
        'filename': 'log.txt'
    }
}
##########################################################################################
# main

def main_test(epoch, path, use_RRC=None,cuda_device_num=None):
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC is not None:
        env_params['RRC_budget'] = 0
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num
    create_logger(**logger_params)
    _print_config()

    tester_params['model_load'] = {
        'path': path,
        'epoch': epoch,
    }

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    # copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    # copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":

    path = model_load_path
    allin = []
    for i in [model_load_epoch]:
        score_optimal, score_student, gap = main_test(i, path)
        # allin.append([score_optimal, score_student, gap])
    # np.savetxt('result.txt', np.array(allin), delimiter=',')

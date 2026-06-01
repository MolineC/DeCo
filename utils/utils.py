
"""
The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import json
import logging
import logging.config
import os
import shutil
import sys
import time
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
b = os.path.abspath('.')
result_folder = b+'/result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


def mask_after_first_mismatch(solution, optimal_solution):
    """
    Args:
        solution: shape [batch, 50], 当前解
        optimal_solution: shape [batch, 50], 最优解
    Returns:
        modified_solution: shape [batch, 50], 修改后的解（第一个不同位置之后设为 -1）
    """
    # 1. 比较两个张量，找到不相等的位置
    mismatch_mask = (solution != optimal_solution)  # [batch, 50], True 表示不同
    
    # 2. 找到第一个不同的位置（按 batch 维度）
    # 使用 argmax 找到第一个 True 的位置（如果没有不同，则返回 0）
    first_mismatch_indices = torch.argmax(mismatch_mask.int(), dim=1)  # [batch]
    
    # 3. 生成掩码：从第一个不同位置开始，后面的位置设为 True
    batch_size, seq_len = solution.shape
    positions = torch.arange(seq_len, device=solution.device).expand(batch_size, -1)  # [batch, 50]
    mask = positions >= first_mismatch_indices.unsqueeze(1)  # [batch, 50]
    
    # 4. 应用掩码，将 solution 的相应位置设为 -1
    modified_solution = solution.clone()
    modified_solution[mask] = -1
    
    return modified_solution

def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)
# /public/home/luof/project/BQ-POMO/CVRP/utils/log_image_style/style_tsp_100.json

def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)

    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']
    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)

def min_max_scale_batch(z):
    # 计算每个样本的最小值和最大值
    min_val = z.min(dim=1, keepdim=True)[0]  # 形状: [batch, 1]
    max_val = z.max(dim=1, keepdim=True)[0]  # 形状: [batch, 1]
    
    # 避免除零
    scaled = (z - min_val) / (max_val - min_val + 1e-8)
    return scaled
def compute_student_constraint_loss(selected_student, state):
    """
    计算模型选择节点（selected_student）的单步约束违反损失
    参数:
        selected_student: 模型选择的当前节点 (B,)，shape: [B]
        state: 环境状态，含：
            - prev_selected_node: 上一步选择的节点 (B,)，shape: [B]
            - current_time: 当前累计时间 (B,)，shape: [B]
            - node_xy: 所有节点坐标 (B, N, 2)，shape: [B, N, 2]
        node_tw: 节点时间窗口 (B, N, 2)，shape: [B, N, 2]，其中node_tw[:, i, 0]是e_i，node_tw[:, i, 1]是l_i
    返回:
        violation_loss: 约束违反损失 (标量)，批次平均后的损失
    """
    B = selected_student.shape[0]  # 批次大小
    N = state.data.shape[1]  # 节点数量

    # 1. 获取上一步节点和当前节点的坐标（用于计算距离）
    # 当前模型选择节点的坐标: [B, 2]
    prev_coords=state.current_coord
    curr_coords = state.node_xy[torch.arange(B), selected_student]

    # 2. 计算旅行时间（欧氏距离作为简化的旅行耗时）
    travel_time = torch.norm(prev_coords - curr_coords, dim=1)  # 每个样本的旅行时间，shape: [B]

    # 3. 计算到达当前节点的时间
    current_time = state.current_time  # 当前累计时间，shape: [B]
    arrival_time = current_time + travel_time  # 到达时间 = 当前时间 + 旅行时间

    # 4. 获取当前节点的时间窗口 [e_i, l_i]
    e_i = state.node_tw[torch.arange(B), selected_student, 0]  # 最早到达时间，shape: [B]
    l_i = state.node_tw[torch.arange(B), selected_student, 1]  # 最晚到达时间，shape: [B]

    # 5. 计算约束违反（仅惩罚迟到，早到等待不惩罚）
    # 到达时间需满足 max(e_i, arrival_time) <= l_i，否则违反
    actual_arrival_time = torch.max(e_i, arrival_time)  # 实际到达时间（早到需等待到e_i）
    violation = torch.max(torch.zeros_like(actual_arrival_time), actual_arrival_time - l_i)  # 迟到量，shape: [B]

    # 6. 计算约束损失（可加非线性惩罚增强严重违反的权重）
    # 例如用平方惩罚放大严重违反：violation **2
    constraint_loss = violation.mean()  # 批次平均损失

    return constraint_loss


def calculate_tsp_redundancy(solutions):
    """
    计算TSP问题采样路径的重复度
    
    参数:
    solutions: 形状为[batch, sample_nums, node_nums]的tensor，包含TSP问题的解
    
    返回:
    包含各种重复度指标的字典
    """
    # 获取输入数据的维度信息
    batch_size, sample_nums, node_nums = solutions.shape
    print(f"数据集信息: 实例数={batch_size}, 每个实例采样数={sample_nums}, 节点数={node_nums}")
    
    # 如果采样数小于2，无法计算重复度
    if sample_nums < 2:
        print("警告: 每个实例的采样数必须至少为2才能计算重复度")
        return None
    
    # 存储每个实例的重复度指标
    exact_duplicates = torch.zeros(batch_size)  # 完全重复的路径对数量
    avg_subpath_similarity = torch.zeros(batch_size)  # 平均子路径相似度
    avg_position_overlap = torch.zeros(batch_size)  # 平均位置重叠度
    
    # 对每个实例计算重复度
    for b in range(batch_size):
        instance_solutions = solutions[b]
        
        # 1. 计算完全重复的路径对
        duplicate_pairs = 0
        total_pairs = 0
        
        # 计算子路径相似度和位置重叠度
        subpath_sims = []
        position_overlaps = []
        
        # 检查所有可能的路径对
        for i, j in combinations(range(sample_nums), 2):
            total_pairs += 1
            path1 = instance_solutions[i]
            path2 = instance_solutions[j]
            
            # 检查路径是否完全相同
            if torch.all(path1 == path2):
                duplicate_pairs += 1
            
            # 计算子路径相似度（连续2节点的边的重合度）
            edges1 = set()
            edges2 = set()
            
            for k in range(node_nums - 1):
                # 为了避免方向影响，将边存储为排序的元组
                edge1 = tuple(sorted((path1[k].item(), path1[k+1].item())))
                edge2 = tuple(sorted((path2[k].item(), path2[k+1].item())))
                edges1.add(edge1)
                edges2.add(edge2)
            
            # 计算边的重合比例
            common_edges = edges1.intersection(edges2)
            edge_similarity = len(common_edges) / max(len(edges1), len(edges2))
            subpath_sims.append(edge_similarity)
            
            # 计算位置重叠度（相同位置出现相同节点的比例）
            position_matches = torch.sum(path1 == path2).item()
            position_overlap = position_matches / node_nums
            position_overlaps.append(position_overlap)
        
        # 存储当前实例的指标
        exact_duplicates[b] = duplicate_pairs / total_pairs if total_pairs > 0 else 0
        avg_subpath_similarity[b] = np.mean(subpath_sims) if subpath_sims else 0
        avg_position_overlap[b] = np.mean(position_overlaps) if position_overlaps else 0
    
    # 计算整体统计指标
    overall_stats = {
        'mean_exact_duplicate_rate': torch.mean(exact_duplicates).item(),
        'std_exact_duplicate_rate': torch.std(exact_duplicates).item(),
        'mean_subpath_similarity': torch.mean(avg_subpath_similarity).item(),
        'std_subpath_similarity': torch.std(avg_subpath_similarity).item(),
        'mean_position_overlap': torch.mean(avg_position_overlap).item(),
        'std_position_overlap': torch.std(avg_position_overlap).item()
    }
    
    # 打印统计结果
    print("\n重复度统计结果:")
    print(f"平均完全重复率: {overall_stats['mean_exact_duplicate_rate']:.4f} (±{overall_stats['std_exact_duplicate_rate']:.4f})")
    print(f"平均子路径相似度: {overall_stats['mean_subpath_similarity']:.4f} (±{overall_stats['std_subpath_similarity']:.4f})")
    print(f"平均位置重叠度: {overall_stats['mean_position_overlap']:.4f} (±{overall_stats['std_position_overlap']:.4f})")
    
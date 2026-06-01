import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pickle
import random

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem)
    finished: torch.Tensor = None
    infeasible: torch.Tensor = None
    # shape: (batch, pomo)
    current_load: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)
    data: torch.Tensor = None
    # shape: (batch, problem, 4)
    node_xy: torch.Tensor = None
    node_dl: torch.Tensor = None
    cost_mat: torch.Tensor = None


class BTWEnv:
    def __init__(self, **env_params):
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.load_list = None
        self.infeasibility_list = None
        self.timeout_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.visited_ninf_flag = None
        self.out_of_dl_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.current_load = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)
        self.device = "cuda:0"
        self.nodes = None
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.sub_path = env_params['sub_path']
        self.batch_size = None
        self.problems = None
        self.raw_data_nodes = []
        self.raw_data_tours = []
        self.selected_count = None
        self.selected_node_list = None
        self.selected_student_list = None
        self.episode = None
        # self.BATCH_IDX = torch.arange(self.env_params['test_batch_size'])[:]

    def load_one_batch_problems(self, episode, batch_size):
        self.episode = episode
        self.batch_size = batch_size
        self.problems_batched, self.solution_batched = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[episode:episode + batch_size]
        self.node_xy_batched = self.problems_batched[:, :, 0:2]
        self.node_demand_batched = self.problems_batched[:, :, 2]
        self.node_draft_limit_batched = self.problems_batched[:, :, 3]
        self.node_dl_batched=self.problems_batched[:,:,2:4]
        # self.raw_node_tw=torch.cat([self.raw_node_tw_starts[:,:,None],self.raw_node_tw_ends[:,:,None]],dim=-1)
        
        self.costmat_batched = self.costmat[episode:episode + batch_size]
       
        self.optimal_length_batched = self.optlens[episode:episode + batch_size]
        self.problem_size = self.problems_batched.shape[1]
    

    def shuffle_data(self):
        mask = torch.randperm(len(self.raw_node_xy)).long()
        self.raw_node_xy = self.raw_node_xy[mask]
        self.raw_node_demand = self.raw_node_demand[mask]
        self.raw_node_draft_limit = self.raw_node_draft_limit[mask]
        self.raw_data_nodes = self.raw_data_nodes[mask]
        self.raw_data_tours = self.raw_data_tours[mask]
        self.optlens = self.optlens[mask]

    def load_solution(self, filename, disable_print=False):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if not disable_print:
            print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))

        return data

    def load_tw_dataset(self, path, solution_path, offset=0, expand=False, num_samples=100000, print_info=True):

        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]
            if print_info:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy,  node_demand, node_draft_limit = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data]
        self.raw_node_xy, self.raw_node_demand, self.raw_node_draft_limit = torch.Tensor(
            node_xy),  torch.Tensor(node_demand), torch.Tensor(node_draft_limit)

        costmat = torch.cdist(self.raw_node_xy[:, None, 0], self.raw_node_xy[:, 1:],
                              compute_mode='donot_use_mm_for_euclid_dist').squeeze(1)
        self.raw_node_draft_limit[:, 0] = (costmat + self.raw_node_draft_limit[:, 1:]).max(dim=-1)[0]

        self.costmat = torch.cdist(self.raw_node_xy, self.raw_node_xy, p=2)

        feature = torch.cat((self.raw_node_xy, self.raw_node_demand[:, :, None], self.raw_node_draft_limit[:, :, None]),
                            dim=2)
        solution_data = self.load_solution(solution_path)
        solution_data = solution_data[offset: offset + num_samples]
        float_values = [item[0]  for item in solution_data]  # 
        int_sequences = [item[1] for item in solution_data]  # 
        merged_array = np.array(int_sequences)

        tours = torch.tensor(merged_array, dtype=torch.int64, requires_grad=False)  #
        tours = torch.cat([
            torch.zeros((tours.size(0), 1), dtype=torch.int64),  
            tours
        ], dim=1)
        solution = tours
        self.optlens = torch.tensor(float_values, dtype=torch.float32, requires_grad=False)  #
        # feature shape(batch,problems,4)
        self.raw_data_nodes = feature
        self.raw_data_tours = tours



        if expand:
            self.raw_data_nodes = self.augment_xy_data_by_8_fold(feature)
            self.raw_data_tours = tours.repeat(8, 1)
            # self.episode=self.episode*8


        # return self.raw_data_nodes,self.raw_data_tours,self.raw_node_service_times,self.raw_node_demand,self.raw_node_draft_limit
        return self.optlens, feature, solution
    def load_all_dataset(self, path, solution_path, offset=0, expand=False, num_samples=100000, print_info=True):

        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        Data_list=["./dldata/tspdl50_hard_10w.pkl","./dldata/tspdl50_medium_10w.pkl"]
        Solution_list=["./dldata/lkh_tspdl50_hard_10w.pkl","./dldata/lkh_tspdl50_medium_10w.pkl"]
        data=[]
        for path in Data_list:
            with open(path, 'rb') as f:
                temp_data = pickle.load(f)[0: offset+num_samples]
                if print_info:
                    print(">> Load {} data ({}) from {}".format(len(temp_data), type(temp_data), path))
                data=data+temp_data
        indices = list(range(len(data)))

        random.shuffle(indices)
        data = [data[i] for i in indices]
        node_xy,  node_demand, node_draft_limit = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data]
        self.raw_node_xy, self.raw_node_demand, self.raw_node_draft_limit = torch.Tensor(
            node_xy),  torch.Tensor(node_demand), torch.Tensor(node_draft_limit)

        costmat = torch.cdist(self.raw_node_xy[:, None, 0], self.raw_node_xy[:, 1:],
                              compute_mode='donot_use_mm_for_euclid_dist').squeeze(1)
        self.raw_node_draft_limit[:, 0] = (costmat + self.raw_node_draft_limit[:, 1:]).max(dim=-1)[0]

        self.costmat = torch.cdist(self.raw_node_xy, self.raw_node_xy, p=2)

        feature = torch.cat((self.raw_node_xy, self.raw_node_demand[:, :, None], self.raw_node_draft_limit[:, :, None]),
                            dim=2)
        solution_data=[]
        for solution_path in Solution_list:    
            temp_data=self.load_solution(solution_path) 
            solution_data=solution_data+temp_data
        solution_data = [solution_data[i] for i in indices]
        float_values = [item[0]  for item in solution_data]  # 
        int_sequences = [item[1] for item in solution_data]  # 
        merged_array = np.array(int_sequences)

        tours = torch.tensor(merged_array, dtype=torch.int64, requires_grad=False)  #
        tours = torch.cat([
            torch.zeros((tours.size(0), 1), dtype=torch.int64),  # 
            tours
        ], dim=1)
        solution = tours
        self.optlens = torch.tensor(float_values, dtype=torch.float32, requires_grad=False)  #
        # 
        # 
        self.raw_data_nodes = feature
        self.raw_data_tours = tours



        if expand:
            self.raw_data_nodes = self.augment_xy_data_by_8_fold(feature)
            self.raw_data_tours = tours.repeat(8, 1)
            # self.episode=self.episode*8
        # 输入数据和标签

        # return self.raw_data_nodes,self.raw_data_tours,self.raw_node_service_times,self.raw_node_demand,self.raw_node_draft_limit
        return self.optlens, feature, solution
    def augment_xy_data_by_8_fold(self, problems):
        # problems.shape: (batch, problem, 4)
        tw = problems[:, :, 2:4]
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]
        # x,y shape: (batch, problem, 1)

        dat1 = torch.cat((x, y, tw), dim=2)
        dat2 = torch.cat((1 - x, y, tw), dim=2)
        dat3 = torch.cat((x, 1 - y, tw), dim=2)
        dat4 = torch.cat((1 - x, 1 - y, tw), dim=2)
        dat5 = torch.cat((y, x, tw), dim=2)
        dat6 = torch.cat((1 - y, x, tw), dim=2)
        dat7 = torch.cat((y, 1 - x, tw), dim=2)
        dat8 = torch.cat((1 - y, 1 - x, tw), dim=2)

        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, problem, 4)

        return aug_problems
    def augment_xy_data_by_16_fold(self, problems):
        # problems.shape: (batch, problem, 4)
        tw = problems[:, :, 2:4]
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]
        # x,y shape: (batch, problem, 1)

        dat1 = torch.cat((x, y, tw), dim=2)
        dat2 = torch.cat((1 - x, y, tw), dim=2)
        dat3 = torch.cat((x, 1 - y, tw), dim=2)
        dat4 = torch.cat((1 - x, 1 - y, tw), dim=2)
        dat5 = torch.cat((y, x, tw), dim=2)
        dat6 = torch.cat((1 - y, x, tw), dim=2)
        dat7 = torch.cat((y, 1 - x, tw), dim=2)
        dat8 = torch.cat((1 - y, 1 - x, tw), dim=2)

        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        aug_problems=torch.cat((aug_problems,aug_problems),dim=0)
        # shape: (8*batch, problem, 4)

        return aug_problems
    def augment_xy_data(self, problems, target_size):
        # 1. 基础数据提取
        tw = problems[:, :, 2:4]
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]

        base_augs = [
            torch.cat((x, y, tw), dim=2),           # 
            torch.cat((1 - x, y, tw), dim=2),       # 
            torch.cat((x, 1 - y, tw), dim=2),       # 
            torch.cat((1 - x, 1 - y, tw), dim=2),   # 
            torch.cat((y, x, tw), dim=2),           # 
            torch.cat((1 - y, x, tw), dim=2),       # 
            torch.cat((y, 1 - x, tw), dim=2),       # 
            torch.cat((1 - y, 1 - x, tw), dim=2)    # 
        ]
        
        aug_problems = torch.cat(base_augs, dim=0)

        if target_size > 8:
            repeat_times = target_size // 8
            aug_problems = aug_problems.repeat(repeat_times, 1, 1)

        return aug_problems
    def augment_data(self, sample_size=8):
        
        self.problems_batched=self.augment_xy_data(self.problems_batched,sample_size)
        self.node_xy_batched=self.problems_batched[:,:,0:2]
        self.node_demand_batched = self.problems_batched[:, :, 2]
        self.node_draft_limit_batched = self.problems_batched[:, :, 3]
        self.node_dl_batched=self.problems_batched[:,:,2:4]
        self.solution_batched = self.solution_batched.repeat(sample_size,1)  # 
        self.optimal_length_batched = self.optimal_length_batched.repeat(sample_size)
    def sample_data(self, sample_size):
        #
        self.problems_batched = torch.repeat_interleave(self.problems_batched, repeats=sample_size, dim=0)
        self.node_xy_batched=self.problems_batched[:,:,0:2]
        self.node_demand_batched = self.problems_batched[:, :, 2]
        self.node_draft_limit_batched = self.problems_batched[:, :, 3]
        self.node_dl_batched=self.problems_batched[:,:,2:4]
        self.solution_batched=torch.repeat_interleave(self.solution_batched, repeats=sample_size, dim=0)
        self.optimal_length_batched=torch.repeat_interleave(self.optimal_length_batched, repeats=sample_size, dim=0)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long).to(self.device)
        self.load_list = torch.zeros((self.batch_size, 0)).to(self.device)
        self.timeout_list = torch.zeros((self.batch_size, 0)).to(self.device)
        self.length_list = torch.zeros((self.batch_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, 0), dtype=torch.bool).to(
            self.device)  # True for causing infeasibility
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size , self.problem_size)).to(self.device)
        self.out_of_dl_ninf_flag = torch.zeros(size=(self.batch_size, self.problem_size)).to(self.device)
        
        self.current_load = self.node_demand_batched.sum(dim=1).to(self.device)
        self.infeasible = torch.zeros(self.batch_size, dtype=torch.bool).to(self.device)
        # self.current_load = torch.zeros(self.batch_size).to(self.device)
        self.length = torch.zeros(self.batch_size).to(self.device)

        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        self.step_state = Step_State(data=self.problems_batched, node_dl=self.node_dl_batched,
                                     node_xy=self.node_xy_batched)
        # self.beam_size =1
        self.BATCH_IDX = torch.arange(self.batch_size)[:]
        reward = None
        done = False

        return None, reward, done

    

    def pre_step(self):
        done = False
        return self.step_state, None, None, done
    def step_test(self, selected_teacher, selected_student):
        # 
        self.selected_count += 1
        self.current_node = selected_teacher

        # --- Infeasible Check ---
        newly_infeasible = self.out_of_dl_ninf_flag[
            torch.arange(self.batch_size)[:, None], 
            selected_teacher[:, None].reshape(self.batch_size, 1)
        ] == float('-inf')
        self.infeasible = self.infeasible + newly_infeasible.sum(dim=-1)

        # 
        self.visited_ninf_flag[self.BATCH_IDX, selected_teacher] = float('-inf')
        self.selected_node_list = torch.cat((self.selected_node_list, selected_teacher[:, None]), dim=1)
        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        # 2. 
        current_coord = self.node_xy_batched[torch.arange(self.batch_size)[:], selected_teacher]

        # 
        if self.selected_count > 1:
            new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
            self.length = self.length + new_length
        
        # 
        self.current_coord = current_coord

        # 
        node_demand = self.node_demand_batched[torch.arange(self.batch_size), selected_teacher]
        self.current_load = self.current_load - node_demand

        # 
        if self.selected_count > 0:
            arrival_load_next = self.current_load[:, None] 
            out_of_draft = arrival_load_next > self.node_draft_limit_batched
            self.out_of_dl_ninf_flag = torch.zeros(size=(self.batch_size, self.problem_size)).to(self.current_load.device)
            self.out_of_dl_ninf_flag[out_of_draft] = float('-inf')

        # 
        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:, None]), dim=1)
        self.load_list = torch.cat((self.load_list, self.current_load[:, None]), dim=1)
        self.length_list = torch.cat((self.length_list, self.length[:, None]), dim=1)
        
        done = (self.selected_count == self.problems_batched.shape[1])

        # 更新 Step State
        self.step_state.current_load = self.current_load
        self.step_state.length = self.length

        return self.step_state, self.infeasibility_list, done, self.infeasible, self.visited_ninf_flag

    def make_dir(self, path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def drawPic(self, arr_, tour_, name='xx', optimal_tour_=None, index=None):
        arr = arr_[index.item()].clone().cpu().numpy()
        tour = tour_[index.item()].clone().cpu().numpy()
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        arr = (arr - arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20))

        plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

        plt.axis('off')

        start = [arr[tour[0], 0], arr[tour[-1], 0]]
        end = [arr[tour[0], 1], arr[tour[-1], 1]]
        plt.plot(start, end, color='red', linewidth=2, )

        for i in range(len(tour) - 1):
            tour = np.array(tour, dtype=int)
            start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
            end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
            plt.plot(start, end, color='red', linewidth=2)

        b = os.path.abspath(".")
        path = b + '/figure'
        self.make_dir(path)
        plt.savefig(path + f'/{name}.pdf', bbox_inches='tight', pad_inches=0)

    def _get_travel_distance(self):

        gathering_index = self.solution_batched.unsqueeze(2).expand(self.batch_size, self.problems_batched.shape[1], 2)
        seq_expanded = self.problems_batched
        ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2)
        segment_lengths = segment_lengths.sum(2).sqrt()
        travel_distances = segment_lengths.sum(1)

        # trained model's distance
        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(-1, self.problems_batched.shape[1], 2)
        ordered_seq_student = self.problems_batched.gather(dim=1, index=gathering_index_student)
        rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
        segment_lengths_student = ((ordered_seq_student - rolled_seq_student) ** 2)
        segment_lengths_student = segment_lengths_student.sum(2).sqrt()
        # shape: (batch,problem)
        travel_distances_student = segment_lengths_student.sum(1)
        # shape: (batch)
        return travel_distances, travel_distances_student

    def _get_travel_distance_2(self, problems, solution):

        gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)

        seq_expanded = problems

        ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

        segment_lengths = ((ordered_seq - rolled_seq) ** 2)

        segment_lengths = segment_lengths.sum(2).sqrt()

        travel_distances = segment_lengths.sum(1)

        return travel_distances
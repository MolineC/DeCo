import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
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
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)
    data: torch.Tensor = None
    # shape: (batch, problem, 4)
    node_xy: torch.Tensor = None
    node_tw: torch.Tensor = None
    cost_mat: torch.Tensor = None


class BTWEnv:
    def __init__(self, **env_params):
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.timestamps = None
        self.infeasibility_list = None
        self.timeout_list = None
        # shape: (batch, pomo, 0~)
        # self.is_aug=True
        self.is_aug=False
        # Dynamic-2
        ####################################
        self.visited_ninf_flag = None
        self.out_of_tw_ninf_flag = None
        self.simulated_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.speed=1
        self.current_time = None
        self.out_of_tw_list = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)
        self.device="cuda:0"
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

        # 读取batched-problems
        self.problems_batched, self.solution_batched = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[episode:episode + batch_size]
        self.node_xy_batched=self.problems_batched[:,:,0:2]
        self.node_tw_batched=self.problems_batched[:,:,2:4]
        self.node_tw_start_batched=self.problems_batched[:,:,2]
        self.node_tw_end_batched=self.problems_batched[:,:,3]
        self.node_service_time_batched=self.raw_node_service_times[episode:episode + batch_size]
        self.optimal_length_batched=self.optlens[episode:episode + batch_size]
        # self.costmat_batched=self.costmat[episode:episode + batch_size]

        norw=0
        self.problem_size = self.problems_batched.shape[1]
        tw_end_max =  self.node_tw_end_batched[:, :1]
        if norw:
            tw_end_max =  self.node_tw_end_batched[:, :1]
            self.node_tw_start_batched = self.node_tw_start_batched / tw_end_max
            self.node_tw_end_batched = self.node_tw_end_batched / tw_end_max
            self.node_tw_batched=torch.cat([self.node_tw_start_batched[:,:,None],self.node_tw_end_batched[:,:,None]],dim=-1)
            self.problems_batched=torch.cat([self.node_xy_batched,self.node_tw_start_batched[:,:,None],self.node_tw_end_batched[:,:,None]],dim=-1)
            self.tw_end_max=tw_end_max.squeeze(1)
    def augment_xy_data(self, problems, target_size):

        tw = problems[:, :, 2:4]
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]

        base_augs = [
            torch.cat((x, y, tw), dim=2),           #
            torch.cat((1 - x, y, tw), dim=2),       
            torch.cat((x, 1 - y, tw), dim=2),      
            torch.cat((1 - x, 1 - y, tw), dim=2),   
            torch.cat((y, x, tw), dim=2),           
            torch.cat((1 - y, x, tw), dim=2),       
            torch.cat((y, 1 - x, tw), dim=2),       
            torch.cat((1 - y, 1 - x, tw), dim=2)   
        ]
        aug_problems = torch.cat(base_augs, dim=0)

        if target_size > 8:
            repeat_times = target_size // 8
            aug_problems = aug_problems.repeat(repeat_times, 1, 1)

        return aug_problems
    def augment_xy_data_by_8_fold(self,problems):
        # problems.shape: (batch, problem, 4)
        tw=problems[:,:,2:4]
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]
        # x,y shape: (batch, problem, 1)

        dat1 = torch.cat((x, y,tw), dim=2)
        dat2 = torch.cat((1 - x, y,tw), dim=2)
        dat3 = torch.cat((x, 1 - y,tw), dim=2)
        dat4 = torch.cat((1 - x, 1 - y,tw), dim=2)
        dat5 = torch.cat((y, x,tw), dim=2)
        dat6 = torch.cat((1 - y, x,tw), dim=2)
        dat7 = torch.cat((y, 1 - x,tw), dim=2)
        dat8 = torch.cat((1 - y, 1 - x,tw), dim=2)

        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, problem, 4)

        return aug_problems
   
    def sample_data(self, sample_size):
        # 重复sample_size
        self.sample_size=sample_size
        self.problems_batched = torch.repeat_interleave(self.problems_batched, repeats=sample_size, dim=0)
        self.node_xy_batched=self.problems_batched[:,:,0:2]
        self.node_tw_start_batched=self.problems_batched[:,:,2]
        self.node_tw_batched=self.problems_batched[:,:,2:4]
        self.node_tw_end_batched=self.problems_batched[:,:,3]
        self.node_service_time_batched=torch.repeat_interleave(self.node_service_time_batched, repeats=sample_size, dim=0)
        self.solution_batched=torch.repeat_interleave(self.solution_batched, repeats=sample_size, dim=0)
        self.optimal_length_batched=torch.repeat_interleave(self.optimal_length_batched, repeats=sample_size, dim=0)
    def augment_data(self, sample_size=8):
            # 重复8次
            self.problems_batched=self.augment_xy_data(self.problems_batched,sample_size)
            self.node_xy_batched=self.problems_batched[:,:,0:2]
            self.node_tw_start_batched=self.problems_batched[:,:,2]
            self.node_tw_batched=self.problems_batched[:,:,2:4]
            self.node_tw_end_batched=self.problems_batched[:,:,3]
            self.node_service_time_batched=self.node_service_time_batched.repeat(sample_size,1)
            self.solution_batched=self.solution_batched.repeat(sample_size,1)
            self.optimal_length_batched=self.optimal_length_batched.repeat(sample_size,1)
            self.optimal_length_batched= self.optimal_length_batched.reshape(self.problems_batched.shape[0])
    def augment_16_data(self, sample_size=16):
            # 重复8次
            self.problems_batched=self.augment_xy_data_by_16_fold(self.problems_batched)
            self.node_xy_batched=self.problems_batched[:,:,0:2]
            self.node_tw_start_batched=self.problems_batched[:,:,2]
            self.node_tw_batched=self.problems_batched[:,:,2:4]
            self.node_tw_end_batched=self.problems_batched[:,:,3]

            self.node_service_time_batched=self.node_service_time_batched.repeat(sample_size,1)
            self.solution_batched=self.solution_batched.repeat(sample_size,1)
            self.optimal_length_batched=self.optimal_length_batched.repeat(sample_size,1)
            self.optimal_length_batched= self.optimal_length_batched.reshape(self.problems_batched.shape[0])
    def shuffle_data(self):
        mask = torch.randperm(len(self.raw_node_xy)).long()   
        self.raw_node_xy = self.raw_node_xy[mask]
        self.raw_node_service_times = self.raw_node_service_times[mask]
        self.raw_node_tw_starts = self.raw_node_tw_starts[mask]
        self.raw_node_tw_ends = self.raw_node_tw_ends[mask]
        self.raw_data_nodes = self.raw_data_nodes[mask]
        self.raw_data_tours = self.raw_data_tours[mask]
        self.optlens = self.optlens[mask]
    
    def load_solution(self,filename, disable_print=False):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if not disable_print:
            print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
        
        return data
    
    def load_tw_dataset(self, path, solution_path,offset=0,expand=True, num_samples=100000, print_info=True):
        
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if print_info:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, service_time, node_tw_start, node_tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
        self.raw_node_xy, self.raw_node_service_times, self.raw_node_tw_starts, self.raw_node_tw_ends = torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(node_tw_start), torch.Tensor(node_tw_end)

        loc_factor = 100
        self.raw_node_xy = self.raw_node_xy / loc_factor  # Normalize
        self.raw_node_tw_starts = self.raw_node_tw_starts / (loc_factor)
        self.raw_node_tw_ends = self.raw_node_tw_ends / (loc_factor)
        costmat=torch.cdist(self.raw_node_xy[:, None, 0], self.raw_node_xy[:, 1:],compute_mode='donot_use_mm_for_euclid_dist').squeeze(1) 
        self.raw_node_tw_ends[:, 0] = (costmat+ self.raw_node_tw_ends[:, 1:]).max(dim=-1)[0]
       
        # self.costmat = torch.cdist(self.raw_node_xy, self.raw_node_xy, p=2) 
        self.raw_node_tw=torch.cat([self.raw_node_tw_starts[:,:,None],self.raw_node_tw_ends[:,:,None]],dim=-1)
        
        
        
        feature =  torch.cat((self.raw_node_xy, self.raw_node_tw_starts[:, :, None], self.raw_node_tw_ends[:, :, None]), dim=2)
        solution_data=self.load_solution(solution_path) 
        solution_data=solution_data[offset: offset+num_samples]
        float_values = [item[0]/100 for item in solution_data]  # 提取最优解
        int_sequences = [item[1] for item in solution_data]  # 提取路径
        merged_array = np.array(int_sequences)

        tours = torch.tensor(merged_array, dtype=torch.int64, requires_grad=False)   # 
        tours = torch.cat([
            torch.zeros((tours.size(0), 1), dtype=torch.int64), 
            tours
        ], dim=1)
        solution=tours
        # self.optlens=read_dumas_opt_batch(instance_name)
       
        self.optlens = torch.tensor(float_values, dtype=torch.float32, requires_grad=False)   # 
        # 转换为PyTorch张量
        # feature shape(batch,problems,4)
        self.raw_data_nodes = feature
        self.raw_data_tours = tours
      


        if expand:
            self.raw_data_nodes=self.augment_xy_data_by_8_fold(feature)
            self.raw_data_tours=tours.repeat(8,1)
            self.raw_node_service_times=self.raw_node_service_times.repeat(8,1) 

        return self.optlens,feature,solution
    
    def load_all_dataset(self, path, solution_path,offset=0,expand=True, num_samples=100000, print_info=True):
        
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        # easy.medium.hard
        
        Data_list=["./twdata/tsptw50_hard_10w.pkl","./twdata/tsptw50_medium_10w.pkl","./twdata/tsptw50_easy_10w.pkl"]
        Solution_list=["./twdata/lkh_tsptw50_hard_10w.pkl","./twdata/lkh_tsptw50_medium_10w.pkl","./twdata/lkh_tsptw50_easy_10w.pkl"]
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
        node_xy, service_time, node_tw_start, node_tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
        self.raw_node_xy, self.raw_node_service_times, self.raw_node_tw_starts, self.raw_node_tw_ends = torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(node_tw_start), torch.Tensor(node_tw_end)

        # Normalize as in DPDP (Kool et. al)
        loc_factor = 100
        self.raw_node_xy = self.raw_node_xy / loc_factor  # Normalize
        self.raw_node_tw_starts = self.raw_node_tw_starts / loc_factor
        self.raw_node_tw_ends = self.raw_node_tw_ends / loc_factor
        costmat=torch.cdist(self.raw_node_xy[:, None, 0], self.raw_node_xy[:, 1:],compute_mode='donot_use_mm_for_euclid_dist').squeeze(1) 
        self.raw_node_tw_ends[:, 0] = (costmat+ self.raw_node_tw_ends[:, 1:]).max(dim=-1)[0]
       
        self.raw_node_tw=torch.cat([self.raw_node_tw_starts[:,:,None],self.raw_node_tw_ends[:,:,None]],dim=-1)
        
        
        
        feature =  torch.cat((self.raw_node_xy, self.raw_node_tw_starts[:, :, None], self.raw_node_tw_ends[:, :, None]), dim=2)
        # 读取解
        solution_data=[]
        for solution_path in Solution_list:    
            temp_data=self.load_solution(solution_path) 
            temp_data=temp_data[offset: offset+num_samples]
            solution_data=solution_data+temp_data
        solution_data = [solution_data[i] for i in indices]
        float_values = [item[0]/100 for item in solution_data]  # 提取最优解
        int_sequences = [item[1] for item in solution_data]  # 提取路径
        merged_array = np.array(int_sequences)

        tours = torch.tensor(merged_array, dtype=torch.int64, requires_grad=False)   # 
        tours = torch.cat([
            torch.zeros((tours.size(0), 1), dtype=torch.int64),  # 添加首列0,为了符合LEHD的训练范式
            tours
        ], dim=1)
        solution=tours
        self.optlens = torch.tensor(float_values, dtype=torch.float32, requires_grad=False)   # 
        # 转换为PyTorch张量
        # feature shape(batch,problems,4)
        self.raw_data_nodes = feature
        self.raw_data_tours = tours
        return self.optlens,feature,solution
 
    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size , 0), dtype=torch.long).to(self.device)
        self.timestamps = torch.zeros((self.batch_size , 0)).to(self.device)
        self.timeout_list = torch.zeros((self.batch_size , 0)).to(self.device)
        self.length_list=torch.zeros((self.batch_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size , 0), dtype=torch.bool).to(self.device) # True for causing infeasibility
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size , self.problem_size)).to(self.device)
        self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size , self.problem_size)).to(self.device)
        
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size , self.problem_size)).to(self.device)
        self.out_of_tw_list = torch.zeros((self.batch_size, 0)).to(self.device)
        self.infeasible = torch.zeros(self.batch_size , dtype=torch.bool).to(self.device)
        self.current_time = torch.zeros(self.batch_size ).to(self.device)
        if self.is_aug:
            random_time = torch.randint(low=0, high=100, size=(self.batch_size, 1))/100
            # random_time = torch.randint(low=0, high=int(self.node_tw_batched[:,:1,1].squeeze().max()), size=(self.batch_size, 1))
            random_time = random_time.unsqueeze(1)  # 形状变为[batch, 1, 1]
            self.node_tw_batched = self.node_tw_batched+ random_time  # 形状保持[batch, problem_nums, 2]
        self.length = torch.zeros(self.batch_size).to(self.device)
   
        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        self.step_state = Step_State(data=self.problems_batched,node_tw=self.node_tw_batched,node_xy=self.node_xy_batched)
        # self.beam_size =1
        self.BATCH_IDX = torch.arange(self.batch_size)[:]
        reward = None
        done = False
       
        return None, reward, done
   
    def pre_step(self):
        done = False
        return self.step_state, None,None, done
    
    def step_test(self,selected_teacher,selected_student):
        # (batch,)
        ####################################
        self.selected_count += 1
        self.current_node = selected_teacher
        # (batch)
        newly_infeasible=self.out_of_tw_ninf_flag[torch.arange(self.batch_size)[:,None], selected_teacher[:, None].reshape(self.batch_size, 1)]==float('-inf')

        newly_infeasible=newly_infeasible.sum(dim=-1)
        self.infeasible = self.infeasible + newly_infeasible

        self.visited_ninf_flag[self.BATCH_IDX, selected_teacher] = float('-inf')
        self.selected_node_list = torch.cat((self.selected_node_list, selected_teacher[:, None]), dim=1)
        # 学生模型选择的路径，
        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)
        # (batch,0~current_size)
        #  2  计算下一步的mask
        ####################################
        current_coord = self.node_xy_batched[torch.arange(self.batch_size)[:], selected_teacher]
        if self.selected_count >1:
            # shape: (batch, 2)
            new_length = ((current_coord - self.current_coord).norm(p=2, dim=-1))
            self.length = self.length + new_length
        self.current_coord = current_coord
        self.step_state.current_coord=current_coord

        if self.selected_count > 1:
            self.current_time = (torch.max(self.current_time + new_length / self.speed,                                   self.node_tw_start_batched[torch.arange(self.batch_size), selected_teacher])+ self.node_service_time_batched[torch.arange(self.batch_size), selected_teacher])
            round_error_epsilon = 0.0001
            self.step_state.current_time=self.current_time
            self.step_state.length=self.length
            next_arrival_time = torch.max(self.current_time[:,  None] + (self.current_coord[:,  None, :] - self.node_xy_batched[:,  :].expand(-1 ,-1, -1)).norm(p=2, dim=-1) / self.speed,
                                        self.node_tw_start_batched[:, :].expand(-1 ,-1))
            out_of_tw = next_arrival_time > self.node_tw_end_batched[:, :].expand(-1 ,-1) + round_error_epsilon
            self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size , self.problem_size))
            self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')
        else:
            self.step_state.current_time=torch.zeros(self.batch_size)

        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:,None]), dim=1)
        self.timestamps=torch.cat((self.timestamps, self.current_time[:,None]), dim=1)
        self.length_list=torch.cat((self.length_list, self.length[:,None]), dim=1)
        done = (self.selected_count==self.problems_batched.shape[1])
        infeasible = self.infeasible
        return self.step_state, self.infeasibility_list, done, infeasible,self.visited_ninf_flag
    def step_train(self,selected_teacher,selected_student):
        # (batch,)
        ####################################
        self.selected_count += 1
        self.current_node = selected_teacher
        # (batch)

        newly_infeasible=self.out_of_tw_ninf_flag[torch.arange(self.batch_size)[:,None], selected_teacher[:, None].reshape(self.batch_size, 1)]==float('-inf')

        newly_infeasible=newly_infeasible.sum(dim=-1)
        self.infeasible = self.infeasible + newly_infeasible

        self.visited_ninf_flag[self.BATCH_IDX, selected_teacher] = float('-inf')
 
        self.selected_node_list = torch.cat((self.selected_node_list, selected_teacher[:, None]), dim=1)

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)
        # (batch,0~current_size)

        ####################################

        current_coord = self.node_xy_batched[torch.arange(self.batch_size)[:], selected_teacher]

        if self.selected_count >2:
            # shape: (batch, 2)
            new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)

            self.length = self.length + new_length
        self.current_coord = current_coord
       

        if self.selected_count > 2:
            self.current_time = (torch.max(self.current_time + new_length / self.speed,
                                        self.node_tw_start_batched[torch.arange(self.batch_size), selected_teacher])
                                + self.node_service_time_batched[torch.arange(self.batch_size), selected_teacher])
            round_error_epsilon = 0.0001
            self.step_state.current_time=self.current_time
            self.step_state.length=self.length

            next_arrival_time = torch.max(self.current_time[:,  None] + (self.current_coord[:,  None, :] - self.node_xy_batched[:,  :].expand(-1 ,-1, -1)).norm(p=2, dim=-1) / self.speed,
                                        self.node_tw_start_batched[:, :].expand(-1 ,-1))
            
            out_of_tw = next_arrival_time > self.node_tw_end_batched[:, :].expand(-1 ,-1) + round_error_epsilon
            self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size , self.problem_size))



            self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')
        else:
            self.step_state.current_time=torch.zeros(self.batch_size)

        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:,None]), dim=1)
        self.timestamps=torch.cat((self.timestamps, self.current_time[:,None]), dim=1)
        self.length_list=torch.cat((self.length_list, self.length[:,None]), dim=1)
        done = (self.selected_count==self.problems_batched.shape[1])
        if done:
            infeasible = self.infeasible
        else:
            infeasible = 0
        return self.step_state, self.infeasibility_list, done, infeasible,self.visited_ninf_flag
    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def drawPic(self, arr_, tour_, name='xx',optimal_tour_=None,index=None):
        arr = arr_[index.item()].clone().cpu().numpy()
        tour =  tour_[index.item()].clone().cpu().numpy()
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        arr = (arr -arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20 ))

        plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

        plt.axis('off')

        start = [arr[tour[0], 0], arr[tour[-1], 0]]
        end = [arr[tour[0], 1], arr[tour[-1], 1]]
        plt.plot(start, end, color='red', linewidth=2, )


        for i in range(len(tour) - 1):
            tour = np.array(tour, dtype=int)
            start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
            end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
            plt.plot(start,end,color='red',linewidth=2)

        b = os.path.abspath(".")
        path = b+'/figure'
        self.make_dir(path)
        plt.savefig(path+f'/{name}.pdf',bbox_inches='tight', pad_inches=0)


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
    def _get_part_travel_distance(self, problems, solution):
        batch_size, num_nodes = solution.shape
        
        gathering_index = solution.unsqueeze(2).expand(batch_size, num_nodes, 2)
        
        ordered_seq = problems.gather(dim=1, index=gathering_index)
        
        start_nodes = ordered_seq[:, :-1, :] 

        end_nodes = ordered_seq[:, 1:, :]   

        segment_lengths = ((start_nodes - end_nodes) ** 2).sum(2).sqrt() 
        

        travel_distances = segment_lengths.sum(1) 
        
        return travel_distances
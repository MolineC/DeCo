
from logging import getLogger

import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from DeCo.TSPDL.DeCoModel import TSPModel as Model
from DLEnv import BTWEnv as Env
from utils.utils import *
import copy
import torch.distributions as dist


class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        
        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        # torch.set_printoptions(precision=4)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()

    def _test_one_batch(self, episode, batch_size):
        self.model.eval()
        # 计算时间
        with torch.no_grad():
            tw_norm=self.tester_params['tw_norm']
            if self.tester_params["Sample"]:
                aug_batch_size=self.tester_params["augment_batch_size"]
                self.env.load_one_batch_problems(episode, aug_batch_size)
                self.env.sample_data(self.tester_params["sample_size"])
                sample_size=self.tester_params["sample_size"]
                batch_size=aug_batch_size*sample_size
                Greedy=False
            elif self.tester_params["augment"]:
                aug_batch_size=self.tester_params["augment_batch_size"]
                self.env.load_one_batch_problems(episode, aug_batch_size)
                sample_size=self.tester_params["sample_size"]
                self.env.augment_data(sample_size)
                batch_size=aug_batch_size*sample_size
                Greedy=False
            else:
                self.env.load_one_batch_problems(episode, batch_size)
                Greedy=True
                

            B_V = batch_size * 1
            self.env.batch_size=B_V

            self.env.reset()
            current_step = 0

            state, _, _, done = self.env.pre_step()  # state: data, first_node = current_node

            prob_list = torch.ones(size=(batch_size, 1))
            while not done:
                if current_step == 0:
                    selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                    selected_student = selected_teacher
                else:
                    selected_teacher, prob,probs,selected_student = self.model(state,self.env.selected_node_list,self.env.solution_batched,current_step, tw_norm,tw_mask,Greedy)
                    prob_list = torch.cat((prob_list, prob), dim=1)
                current_step += 1 
                selected_teacher=selected_student
                state, infeasiblelist, done,infeasible,tw_mask = self.env.step_test(selected_teacher, selected_student)# 

            best_select_node_list = self.env.selected_node_list
            timeout=infeasiblelist[:,-1].sum()
            if self.tester_params["Sample"] :       
                print("batch_size",batch_size)
                best_select_node_list=best_select_node_list.reshape(aug_batch_size,sample_size,-1)
                solution_feasible_rate = round(((infeasible==0).sum()/(aug_batch_size*sample_size)).item() * 100, 2)
                print("feasible_solution",(infeasible==0).sum().item() )
                infeasible=infeasible.reshape(aug_batch_size,sample_size)
        
                timeout=infeasiblelist[:,-1].sum()
               
                print("timeout node nums:",timeout.sum().item())
                print("illegal_rate",timeout.sum().item()/(aug_batch_size*sample_size*self.env.problem_size))
                instance_infeasible_rate=1-((infeasible==0).any(dim=-1).sum().item()/(aug_batch_size))
                current_best_length = self.env._get_travel_distance_2(self.env.node_xy_batched, best_select_node_list.reshape(aug_batch_size*sample_size,-1))
                current_best_length=current_best_length.reshape(aug_batch_size,sample_size)
                solution_infeasible_rate=100-solution_feasible_rate
                instance_infeasible_rate=instance_infeasible_rate*100
                print("solution_infeasible_rate:",solution_infeasible_rate,"%")
                print("instance_infeasible_rate:",instance_infeasible_rate,"%")
                self.env.optimal_length_batched=self.env.optimal_length_batched.reshape(aug_batch_size,sample_size)
                feasible_mask=(infeasible==0)
                aug_gap=[]
                aug_score=[]
                fsb_stu=torch.zeros(aug_batch_size)
                for i in range(aug_batch_size):
                    if feasible_mask[i].sum()==0: 
                        continue
                    fsb_stu=current_best_length[i][feasible_mask[i]].min()
                    aug_score.append(fsb_stu)
                    fsb_opt=self.env.optimal_length_batched[i][feasible_mask[i]].min()
                    fsb_gap=((fsb_stu-fsb_opt)/fsb_opt)*100  
                    aug_gap.append(fsb_gap.item())
                if len(aug_gap)==0:
                    print("no feasible solution")
                    gap=float("nan")
                else:
                    gap=sum(aug_gap)/len(aug_gap)
                print("fsb_gap",gap,"%")
               
                score=fsb_stu.mean().item()
            elif self.tester_params["augment"] :       
                    print("batch_size",batch_size)
                    best_select_node_list=best_select_node_list.reshape(sample_size,aug_batch_size,-1)
                    
                    Feasible_solution_nums=(infeasible==0).sum()
                    total_solution_nums=(aug_batch_size*sample_size)
                    solution_feasible_rate = round((Feasible_solution_nums/total_solution_nums).item() * 100, 2)
                    print("feasible_solution",Feasible_solution_nums.item())
                    infeasible=infeasible.reshape(sample_size,aug_batch_size)
            
                    timeout=infeasiblelist[:,-1].sum()
                    print("timeout node nums:",timeout.sum().item())
                    print("illegal_rate",timeout.sum().item()/(aug_batch_size*sample_size*self.env.problem_size))
                    instance_infeasible_rate=1-((infeasible==0).any(dim=0).sum().item()/(aug_batch_size))
                    current_best_length = self.env._get_travel_distance_2(self.env.node_xy_batched, best_select_node_list.reshape(aug_batch_size*sample_size,-1))
                    current_best_length=current_best_length.reshape(sample_size,aug_batch_size)
                    solution_infeasible_rate=100-solution_feasible_rate
                    instance_infeasible_rate=instance_infeasible_rate*100
                    print("solution_infeasible_rate:",solution_infeasible_rate,"%")
                    print("instance_infeasible_rate:",instance_infeasible_rate,"%")
                    self.env.optimal_length_batched=self.env.optimal_length_batched.reshape(sample_size,aug_batch_size)
                    feasible_mask=(infeasible==0)
                    aug_gap=[]
                    aug_score=[]
                    fsb_stu=torch.zeros(batch_size)
                    for i in range(aug_batch_size):
                        if feasible_mask.sum(dim=0)[i]==0: 
                            continue
                        fsb_stu=current_best_length[:,i][feasible_mask[:,i]].min()
                       
                        aug_score.append(fsb_stu)
                        fsb_opt=self.env.optimal_length_batched[:,i][0]
                        fsb_gap=((fsb_stu-fsb_opt)/fsb_opt)*100  
                        aug_gap.append(fsb_gap.item())
                    if len(aug_gap)==0:
                        print("no feasible solution")
                        gap = float("nan")
                    else :
                        gap = sum(aug_gap) / len(aug_gap)
                        print("feasible_sample_min_gap",sum(aug_gap)/len(aug_gap),"%")
                        if sum(aug_gap)/len(aug_gap)<0:
                            print("new !!")
                   
                    gap=sum(aug_gap)/(len(aug_gap)+1e-6)
                    score=fsb_stu.mean().item()
            else:
                timeout=infeasiblelist[:,-1].sum()
                # infeasible_list: [batch,50]
                print("timeout node nums:",timeout.item())
                print("illegal_rate",timeout.sum().item()/(batch_size*self.env.problem_size))

                instance_feasible_rate = (infeasible==0).sum()/(batch_size)
                feasible_mask= (infeasible==0)
                instance_feasible_rate = round(instance_feasible_rate.item() * 100, 2)
                solution_feasible_rate=instance_feasible_rate
                instance_infeasible_rate=100-instance_feasible_rate
                solution_infeasible_rate=100-instance_feasible_rate
                current_best_length = self.env._get_travel_distance_2(self.env.node_xy_batched, best_select_node_list)
                print("solution_infeasible_rate:",solution_infeasible_rate,"%")
                print("instance_infeasible_rate:",instance_infeasible_rate,"%")
                # 
                fsb_opt=self.env.optimal_length_batched[feasible_mask]
                fsb_stu=current_best_length[feasible_mask]
                fsb_gap=((fsb_stu-fsb_opt)/fsb_opt)*100            
                print("fsb_gap",fsb_gap.mean().item(),"%")
                gap=fsb_gap.mean().item()
                score=fsb_stu.mean().item()
           
 

            return  self.env.optimal_length_batched.mean().item(),score,gap,solution_infeasible_rate,instance_infeasible_rate  
    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()
        if (self.tester_params["augment"]):
            self.env.batch_size=self.tester_params["augment_batch_size"]
            self.env.BATCH_IDX = torch.arange(self.tester_params["augment_batch_size"])[:]
        
        self.env.load_tw_dataset(self.env_params['data_path'],self.env_params['solution_path'] )#读取tsptw的

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        gap_Am=AverageMeter()
        solution_inf_rate_AM = AverageMeter()
        instance_inf_rate_AM = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            score_teacher,score_student,aug_gap,solution_inf_rate,instance_inf_rate = self._test_one_batch(episode,batch_size)
               


            if np.isnan(aug_gap):
                aug_gap = gap_Am.avg
            Feasible_batch=batch_size*(1-instance_inf_rate/100)
            gap_Am.update(aug_gap, Feasible_batch)
            score_AM.update(score_teacher, Feasible_batch)
            score_student_AM.update(score_student, Feasible_batch)
            solution_inf_rate_AM.update(solution_inf_rate, batch_size)
            instance_inf_rate_AM.update(instance_inf_rate, batch_size)


            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},aug_gap: {:.4f}%".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher,score_student,aug_gap))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Solution Infeasible Rate: {:.4f}% ".format(solution_inf_rate_AM.avg))
                self.logger.info(" Instance Infeasible Rate: {:.4f}% ".format(instance_inf_rate_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format(gap_Am.avg))
           

        checkpoint_dict = {
                    'epoch': 100,
                    'model_state_dict': self.model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'scheduler_state_dict': self.scheduler.state_dict(),
                } 
        torch.save(checkpoint_dict,'checkpoint-100.pt')
        return score_AM.avg, score_student_AM.avg, gap_Am.avg
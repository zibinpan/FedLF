
import gfedplat as fp
import copy
import math
import numpy as np
import torch
import time
from gfedplat.algorithm.common.utils import get_fedmdfg_d


class FedMDFG(fp.Algorithm):
    def __init__(self,
                 name='FedMDFG',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 test_conflicts=False,
                 params=None,
                 theta=11.25,  
                 s=5,  
                 *args,
                 **kwargs):

        if params is not None:
            theta = params['theta']
            s = params['s']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' theta' + str(theta) + ' s' + str(s)
        
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, metric_list,
                         max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        
        if theta <= 0 or theta >= 90:
            raise RuntimeError('illegal parameter setting.')
        if s < 0:
            raise RuntimeError('illegal parameter setting.')
        self.theta = theta
        self.s = s
        
        self.prefer_active = 0  
        
        self.last_client_id_list = None  
        self.last_g_locals = None  
        self.last_d = None
        self.client_expected_loss = [None] * self.client_num  
        self.client_join_count = [0] * self.client_num  
        self.same_user_flag = True  

    def line_search(self, g_locals, d, prefer_vec, base_lr, l_locals_0, live_idx, scale):
        old_loss_norm = float(torch.sum(l_locals_0))  
        prefer_vec_norm = torch.norm(prefer_vec)
        old_cos = l_locals_0 @ prefer_vec / (torch.norm(l_locals_0) * prefer_vec_norm)  
        beta = 1e-4
        c = -(g_locals@d)
        if self.same_user_flag:
            lr = float(2**self.s * base_lr)
        else:
            g_bias = float(self.last_d @ d / (torch.norm(self.last_d) * torch.norm(d)))
            lr = float(base_lr)

        old_model = copy.deepcopy(self.module.model.state_dict())  
        min_lr = float(0.5**self.s * base_lr / scale)

        
        lr_storage = []
        norm_storage = []
        while lr >= min_lr:
            self.module.model.load_state_dict(old_model)
            self.update_module(self.module, self.optimizer, lr, d)
            
            l_locals = self.get_loss()  
            l_locals = l_locals[live_idx]  
            
            l_locals_norm = float(torch.sum(l_locals))
            lr_storage.append(lr)
            norm_storage.append(l_locals_norm)
            
            
            if self.prefer_active == 0 and torch.all(l_locals_0 - l_locals >= lr * beta * c):
                lr_storage = []
                norm_storage = []
                break
            
            elif self.prefer_active == 1 and torch.all(l_locals_0 - l_locals >= lr * beta * c) and (l_locals @ prefer_vec) / (torch.norm(l_locals) * prefer_vec_norm) - old_cos > 0:
                lr_storage = []
                norm_storage = []
                break
            lr /= 2

        
        if len(norm_storage) > 0:
            
            for idx, l_locals_norm in enumerate(norm_storage):
                lr = lr_storage[idx]
                if lr > base_lr and self.same_user_flag == False:  
                    continue
                if l_locals_norm < old_loss_norm:
                    norm_storage = []
                    break

        if len(norm_storage) > 0:
            best_idx = np.argmin(norm_storage)
            lr = lr_storage[best_idx]
        
        self.module.model.load_state_dict(old_model)
        self.update_module(self.module, self.optimizer, lr, d)

    def train_a_round(self):
        
        self.prefer_active = 0
        
        com_time_start = time.time()
        g_locals, l_locals = self.evaluate()
        com_time_end = time.time()
        
        client_id_list = self.get_clinet_attr('id')  
        force_active = False
        increase_count = 0
        for i, client_id in enumerate(client_id_list):
            if self.client_join_count[client_id] == 0:
                self.client_expected_loss[client_id] = l_locals[i]  
            else:
                if l_locals[i] <= self.client_expected_loss[client_id]:  
                    self.client_expected_loss[client_id] = (self.client_expected_loss[client_id] * self.client_join_count[client_id] + l_locals[i]) / (self.client_join_count[client_id] + 1)  
                else:
                    if l_locals[i] > self.client_expected_loss[client_id]:
                        increase_count += 1
            self.client_join_count[client_id] += 1  
        if increase_count > 0 and increase_count < self.client_num:  
            force_active = True

        
        if self.last_client_id_list is not None:
            add_idx = []
            for idx, last_client_id in enumerate(self.last_client_id_list):
                if last_client_id not in client_id_list:
                    add_idx.append(idx)
            if len(add_idx) > 0:
                add_grads = self.last_g_locals[add_idx, :]
                self.same_user_flag = False
            else:
                add_grads = None
                self.same_user_flag = True
        else:
            add_grads = None
            self.same_user_flag = True

        grad_local_norm = torch.norm(g_locals, dim=1)

        live_idx = torch.where(grad_local_norm > 1e-6)[0]
        if len(live_idx) == 0:
            return
        if len(live_idx) > 0:
            g_locals = g_locals[live_idx, :]
            l_locals = l_locals[live_idx]
            grad_local_norm = torch.norm(g_locals, dim=1)

        miu = torch.mean(grad_local_norm)
        g_locals = g_locals / grad_local_norm.reshape(-1, 1) * miu

        prefer_vec = torch.Tensor([1.0] * len(live_idx)).float().to(self.device)

        cal_time_start = time.time()
        d, vec, p_active_flag, optimal_flag, descent_flag = get_fedmdfg_d(g_locals, l_locals, add_grads, self.theta, prefer_vec, force_active, self.device)
        cal_time_end = time.time()
        self.prefer_active = 1

        weights = torch.Tensor([1 / len(live_idx)] * len(live_idx)).float().to(self.device)
        g_norm = torch.norm(weights @ g_locals)
        d_norm = torch.norm(d)
        min_lr = self.lr

        d_old = copy.deepcopy(d)
        d = d / d_norm * g_norm
        while torch.max(-(vec @ d)) > 1e-6:
            if torch.norm(d) > d_norm * 2:
                d /= 2
            else:
                d = d_old
                break
        scale = torch.norm(d) / torch.norm(d_old)
        self.line_search(g_locals, d, prefer_vec, min_lr, l_locals, live_idx, scale)

        self.current_training_num += 1

        self.last_client_id_list = self.get_clinet_attr('id')
        self.last_client_id_list = [self.last_client_id_list[live_idx[i]] for i in range(len(live_idx))]
        self.last_g_locals = copy.deepcopy(g_locals)
        self.last_d = d

        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time_end - cal_time_start

        if self.test_conflicts:
            self.cal_conflicts(g_locals, d)

    def run(self):
        while not self.terminated():
            self.train_a_round()


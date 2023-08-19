
import gfedplat as fp
import numpy as np
import copy
import time
import torch


class AFL(fp.Algorithm):
    def __init__(self,
                 name='AFL',
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
                 lam=0.01,
                 *args,
                 **kwargs):

        if params is not None:
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' lam' + str(lam)

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.lam = lam

        self.dynamic_lambdas = np.ones(
            self.online_client_num) * 1.0 / self.online_client_num

        self.result_module = copy.deepcopy(self.module)

    def run(self):

        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, l_locals = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()

            g_locals = []
            old_model = self.module.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                g_locals.append(
                    (old_model - m_locals[idx].span_model_params_to_vec()) / self.lr)
            g_locals = torch.stack(g_locals)

            weights = torch.Tensor(
                self.dynamic_lambdas).float().to(self.device)

            d = []
            for i in range(len(self.module.Loc_list)):
                g_locals_layer = g_locals[:, self.module.Loc_list[i]]
                d_layer = weights @ g_locals_layer
                d.append(d_layer)
            d = torch.hstack(d)

            self.update_module(self.module, self.optimizer, self.lr, d)

            self.dynamic_lambdas = [
                lmb_i+self.lam * float(loss_i) for lmb_i, loss_i in zip(self.dynamic_lambdas, l_locals)]
            self.dynamic_lambdas = self.project(self.dynamic_lambdas)

            self.result_module = (self.current_comm_round * self.result_module +
                                  self.module) * (1 / (self.current_comm_round + 1))

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

            if self.test_conflicts:
                self.cal_conflicts(g_locals, d)

    def project(self, p):
        u = sorted(p, reverse=True)
        res = []
        rho = 0
        for i in range(len(p)):
            if (u[i] + (1.0/(i + 1)) * (1 - np.sum(np.asarray(u)[:i+1]))) > 0:
                rho = i + 1
        lamb = (1.0/(rho+1e-6)) * (1 - np.sum(np.asarray(u)[:rho]))
        for i in range(len(p)):
            res.append(max(p[i] + lamb, 0))
        return res


import gfedplat as fp
import torch
import time
import numpy as np


class qFedAvg(fp.Algorithm):
    def __init__(self,
                 name='qFedAvg',
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
                 q=0.1,
                 *args,
                 **kwargs):

        if params is not None:
            q = params['q']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' q' + str(q)

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.q = q

        self.lr = self.train_setting['optimizer'].defaults['lr']

    def run(self):

        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, l_locals = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()

            g_locals = []
            old_model_params = self.module.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                grad = (old_model_params -
                        m_locals[idx].span_model_params_to_vec()) * (1 / self.lr)
                g_locals.append(grad)
            g_locals_mat = torch.stack(g_locals)
            l_locals1 = torch.Tensor(l_locals).float().to(self.device) + 1e-10
            Deltas = g_locals_mat * l_locals1.reshape(-1, 1)**self.q

            hs = self.q * l_locals1**(self.q - 1) * torch.norm(
                g_locals_mat, dim=1)**2 + 1.0 / self.lr * l_locals1**self.q

            self.aggregate(old_model_params, Deltas, hs)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

            if self.test_conflicts:
                d = (old_model_params -
                     self.module.span_model_params_to_vec()) / self.lr
                self.cal_conflicts(g_locals, d)

    def aggregate(self, old_model_params, Deltas, hs):
        denominator = torch.sum(hs)
        scaled_deltas = Deltas / denominator
        updates = torch.sum(scaled_deltas, dim=0)
        new_params = old_model_params - updates

        self.module.reshape_vec_to_model_params(new_params)

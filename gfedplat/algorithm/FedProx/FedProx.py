
import gfedplat as fp
import numpy as np
import torch
import copy
from torch.optim import Optimizer
import time


class FedProx(fp.Algorithm):
    def __init__(self,
                 name='FedProx',
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
                 mu=0.5,
                 *args,
                 **kwargs):

        if params is not None:
            mu = params['mu']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' mu' + str(mu)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(
                i, module, device, train_setting, metric_list, mu) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)

    def run(self):

        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, _ = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()
            old_model = self.module.span_model_params_to_vec()

            self.weight_aggregate(m_locals)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

            if self.test_conflicts:
                g_locals = []
                for idx, client in enumerate(m_locals):
                    g_locals.append(
                        (old_model - m_locals[idx].span_model_params_to_vec()) / self.lr)
                d = (old_model - self.module.span_model_params_to_vec()) / self.lr
                self.cal_conflicts(g_locals, d)


class pgd(Optimizer):
    def __init__(self, params, lr=0.1, mu=0.5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 mu=0.5,
                 *args,
                 **kwargs):
        super().__init__(id, module, device, train_setting, metric_list, *args, **kwargs)
        self.mu = mu

    def train(self, epochs, lr):

        self.upload_module = copy.deepcopy(self.module)

        optimizer = pgd(filter(lambda p: p.requires_grad,
                        self.upload_module.model.parameters()), lr=lr, mu=self.mu)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        loss = self.cal_loss(self.upload_module)

        self.upload_loss = float(loss)

        old_model_parameters = self.module.model.parameters()

        self.upload_module.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = self.upload_module.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.upload_module.model.zero_grad()
                loss.backward()
                optimizer.step(old_model_parameters)


import gfedplat as fp
import numpy as np
import copy
import torch
from torch.autograd import Variable
import time


class Ditto(fp.Algorithm):
    def __init__(self,
                 name='Ditto',
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
                 lam=0.1,
                 *args,
                 **kwargs):

        if params is not None:
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' lam' + str(lam)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(
                i, module, device, train_setting, metric_list, lam) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.lam = lam

    def run(self):

        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, _ = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()
            old_model = self.module.span_model_params_to_vec()
            self.aggregate(m_locals)

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

    def aggregate(self, m_locals):
        model_params = self.module.model.state_dict()

        for m_local in m_locals:
            m_local_params = m_local.model.state_dict()
            for layer in model_params.keys():
                model_params[layer] += (m_local_params[layer] -
                                        model_params[layer]) / self.client_num


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 lam=0,
                 *args,
                 **kwargs):
        super().__init__(id, module, device, train_setting, metric_list, *args, **kwargs)
        self.lam = lam

    def train(self, epochs, lr):

        self.upload_module = copy.deepcopy(self.module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        loss = self.cal_loss(self.upload_module)

        self.upload_loss = float(loss)

        self.upload_module.model.train()

        global_model_params = self.module.span_model_params_to_vec()

        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = self.upload_module.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.upload_module.model.zero_grad()
                loss.backward()
                optimizer.step()

                grad_vec = self.upload_module.span_model_grad_to_vec()
                old_global_model_params = self.upload_module.span_model_params_to_vec()
                eff_grad = grad_vec + self.lam * \
                    (old_global_model_params - global_model_params)
                delta = lr * eff_grad
                model_params = old_global_model_params - delta
                self.upload_module.reshape_vec_to_model_params(model_params)

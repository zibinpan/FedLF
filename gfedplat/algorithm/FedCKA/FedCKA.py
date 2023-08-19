
import gfedplat as fp
import numpy as np
import copy
import torch
from torch.autograd import Variable
import time


class FedCKA(fp.Algorithm):
    def __init__(self,
                 name='FedCKA',
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
                 mu=3,
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


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 mu=3,
                 *args,
                 **kwargs):
        super().__init__(id, module, device, train_setting, metric_list, *args, **kwargs)
        self.mu = mu
        self.previous_trained_module = None

    def cal_CKA(self, K1, K2):
        K1_span = K1.reshape(-1)
        s1 = K1_span @ K2.reshape(-1)
        s2 = K1_span @ K1_span
        if s1 < 1e-6 or s2 < 1e-6:

            CKA_value = torch.Tensor([0.0]).float().to(self.device)[0]
        else:
            CKA_value = s1 / ((s1**0.5) * (s2**0.5))
        return CKA_value

    def cal_CKA_loss(self, global_module, local_module, previous_module):
        CKA_loss = 0
        count = 0
        local_module_params = list(local_module.model.parameters())
        previous_module_params = list(previous_module.model.parameters())
        for idx, g_layer in enumerate(global_module.model.parameters()):
            l_layer = local_module_params[idx]
            pl_layer = previous_module_params[idx]
            if g_layer.data is not None and l_layer.data is not None and pl_layer.data is not None:
                e1 = torch.exp(self.cal_CKA(l_layer.data, g_layer.data))
                e2 = torch.exp(self.cal_CKA(l_layer.data, pl_layer.data))
                CKA_loss += -torch.log(e1 / (e1 + e2))
                count += 1
        CKA_loss = CKA_loss / count
        return CKA_loss

    def train(self, epochs, lr):

        self.upload_module = copy.deepcopy(self.module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        loss = self.cal_loss(self.upload_module)

        self.upload_loss = float(loss)

        self.upload_module.model.train()

        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = self.upload_module.model(batch_x)
                if self.previous_trained_module is not None:
                    loss = self.criterion(out, batch_y) + self.mu * self.cal_CKA_loss(
                        self.module, self.upload_module, self.previous_trained_module)
                else:
                    loss = self.criterion(out, batch_y)

                self.upload_module.model.zero_grad()
                loss.backward()
                optimizer.step()

        self.previous_trained_module = copy.deepcopy(self.upload_module)

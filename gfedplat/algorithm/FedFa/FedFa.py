
import gfedplat as fp
import numpy as np
import torch
import copy
import time


class FedFa(fp.Algorithm):
    def __init__(self,
                 name='FedFa',
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
                 beta=0.5,
                 gamma=0.9,
                 *args,
                 **kwargs):
        if params is not None:
            beta = params['beta']
            gamma = params['gamma']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(
                train_setting['lr_decay']) + ' beta' + str(beta) + ' gamma' + str(gamma)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(
                i, module, device, train_setting, metric_list) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.alpha = 1.0 - beta
        self.beta = beta
        self.gamma = gamma

        self.ms = 0.0 * self.module.span_model_params_to_vec()

        self.online_frequency = torch.Tensor(
            [0.0] * self.client_num).float().to(self.device)

    def cal_frequency(self):
        client_id_list = self.get_clinet_attr('id')
        self.online_frequency[client_id_list] += 1
        f_locals = self.online_frequency[client_id_list] / \
            self.current_comm_round
        return f_locals

    def run(self):

        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, l_locals = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()
            old_model = self.module.span_model_params_to_vec()
            acc_locals = torch.Tensor(self.get_clinet_attr(
                'upload_training_acc')).float().to(self.device)
            freq_locals = self.cal_frequency()

            ACCinf = -torch.log2(1.0 * acc_locals /
                                 torch.sum(acc_locals) + 1e-6)
            Finf = -torch.log2(1 - 1.0 * freq_locals /
                               torch.sum(freq_locals) + 1e-6)
            ACCinf = acc_locals / torch.sum(ACCinf)
            Finf = freq_locals / torch.sum(Finf)

            weights = self.alpha * ACCinf + self.beta * Finf

            aggregate_params = self.weight_aggregate(
                m_locals, weights=weights, update_module=False)
            old_model_params = self.module.span_model_params_to_vec()
            dw = aggregate_params - old_model_params

            self.ms = self.gamma * self.ms + dw

            new_model_params = old_model_params + self.ms
            self.module.reshape_vec_to_model_params(new_model_params)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

            if self.test_conflicts:
                g_locals = []
                for idx, client in enumerate(m_locals):
                    g_locals.append(
                        (old_model - m_locals[idx].span_model_params_to_vec()) / self.lr)
                d = (old_model_params -
                     self.module.span_model_params_to_vec()) / self.lr
                self.cal_conflicts(g_locals, d)


class Client(fp.Client):
    def cal_loss_acc(self, module):
        module.model.train()
        total_loss = 0.0
        with torch.no_grad():
            correct = 0
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = module.model(batch_x)
                loss = self.criterion(out, batch_y)
                loss = float(loss)
                total_loss += loss * batch_y.shape[0]
                correct += fp.Correct.calc(out, batch_y)
            loss = total_loss / self.local_training_number
            acc = 100 * correct / self.local_training_number
        return loss, acc

    def train(self, epochs, lr):

        self.upload_module = copy.deepcopy(self.module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        self.upload_module.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = self.upload_module.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.upload_module.model.zero_grad()
                loss.backward()
                optimizer.step()

        loss, acc = self.cal_loss_acc(self.upload_module)

        self.upload_loss = float(loss)

        self.upload_training_acc = acc

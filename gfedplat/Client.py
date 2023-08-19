
import gfedplat as fp
import numpy as np
import torch
import copy


class Client:

    def __init__(self,
                 id=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 dishonest=None,
                 *args,
                 **kwargs):
        self.id = id
        if module is not None:
            module = module
        self.module = module
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.train_setting = train_setting
        self.metric_list = metric_list
        self.dishonest = dishonest
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.training_batch_num = 0
        self.test_batch_num = 0
        self.sgd_step = train_setting['sgd_step']

        self.metric_history = {'training_loss': [],
                               'test_loss': [],
                               'local_test_number': 0,
                               }
        for metric in self.metric_list:
            self.metric_history[metric.name] = []
            if metric.name == 'correct':
                self.metric_history['test_accuracy'] = []
        self.criterion = self.train_setting['criterion'].to(self.device)

        self.upload_loss = None
        self.upload_module = None
        self.upload_grad = None
        self.upload_training_acc = None

    def update_data(self,
                    id,
                    local_training_data,
                    local_training_number,
                    local_test_data,
                    local_test_number,
                    ):
        self.id = id
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.training_batch_num = len(local_training_data)
        self.test_batch_num = len(local_test_data)

    def free_memory(self):
        self.upload_loss = None
        self.upload_module = None
        self.upload_grad = None
        self.upload_training_acc = None

    def get_message(self, msg):

        return_msg = {}

        if msg['command'] == 'cal_loss':
            self.upload_loss = self.cal_loss(self.module)
            return return_msg
        if msg['command'] == 'cal_gradient_loss':
            lr = msg['lr']
            if self.sgd_step:
                self.cal_gradient_loss_sgd(lr)
            else:
                self.cal_gradient_loss(lr)
            return return_msg
        if msg['command'] == 'train':

            epochs = msg['epochs']
            lr = msg['lr']

            if self.sgd_step:
                self.train_SGD(epochs, lr)
            else:
                self.train(epochs, lr)
            return return_msg
        if msg['command'] == 'free_memory':
            self.free_memory()
            return return_msg
        if msg['command'] == 'test':

            test_module = msg['test_module']

            self.test(test_module)
            return return_msg
        if msg['command'] == 'require_loss':

            return_msg['l_local'] = self.upload_loss
            return return_msg
        if msg['command'] == 'require_gradient_loss':

            return_grad = self.upload_grad
            return_loss = self.upload_loss

            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_grad *= self.dishonest['scaled_update']
                if self.dishonest['zero_update'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random_update'] is not None:
                    n = len(return_grad)
                    grad = torch.randn(n).float().to(self.device)

                    return_grad = grad
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_client_module':

            return_module = self.upload_module
            return_loss = self.upload_loss
            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_module = (return_module - self.module) * \
                        self.dishonest['scaled_update'] + self.module
                if self.dishonest['zero_update'] is not None:
                    return_module = copy.deepcopy(self.module)
                if self.dishonest['random_update'] is not None:
                    model_params_span = return_module.span_model_params_to_vec()
                    n = len(model_params_span)
                    updates = torch.randn(n).float().to(self.device)
                    old_model_params_span = self.module.span_model_params_to_vec()

                    return_module_params = old_model_params_span + updates
                    return_module.reshape_vec_to_model_params(
                        return_module_params)

            return_msg['m_local'] = return_module
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_test_result':
            return_msg['metric_history'] = copy.deepcopy(self.metric_history)
            return return_msg
        if msg['command'] == 'require_attribute_value':
            attr = msg['attr']
            return_msg['attr'] = getattr(self, attr)
            return return_msg

    def cal_loss(self, module):
        module.model.train()
        total_loss = 0.0
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = module.model(batch_x)
                loss = self.criterion(out, batch_y)
                loss = float(loss)
                total_loss += loss * batch_y.shape[0]
            loss = total_loss / self.local_training_number
        return loss

    def cal_gradient_loss(self, lr=0.1):
        self.module.model.train()

        grad_mat = []
        total_loss = 0
        weights = []
        for step, (batch_x, batch_y) in enumerate(self.local_training_data):
            batch_x = fp.Module.change_data_device(batch_x, self.device)
            batch_y = fp.Module.change_data_device(batch_y, self.device)
            weights.append(batch_y.shape[0])

            out = self.module.model(batch_x)
            loss = self.criterion(out, batch_y)
            total_loss += loss * batch_y.shape[0]

            self.module.model.zero_grad()
            loss.backward()
            grad_vec = self.module.span_model_grad_to_vec()
            grad_mat.append(grad_vec)
        loss = total_loss / self.local_training_number
        weights = torch.Tensor(weights).float().to(self.device)
        weights = weights / torch.sum(weights)

        grad_mat = torch.stack([grad_mat[i] for i in range(len(grad_mat))])

        g = weights @ grad_mat

        self.upload_grad = g
        self.upload_loss = float(loss)

    def cal_gradient_loss_sgd(self, lr=0.1):
        self.module.model.train()
        weights = []
        sample_idx = int(np.random.choice(len(self.local_training_data), 1))
        (batch_x, batch_y) = self.local_training_data[sample_idx]
        batch_x = fp.Module.change_data_device(batch_x, self.device)
        batch_y = fp.Module.change_data_device(batch_y, self.device)
        weights.append(batch_y.shape[0])

        out = self.module.model(batch_x)
        loss = self.criterion(out, batch_y)

        self.module.model.zero_grad()
        loss.backward()
        self.upload_grad = self.module.span_model_grad_to_vec()
        self.upload_loss = float(loss)

    def train_SGD(self, epochs, lr):
        self.upload_module = copy.deepcopy(self.module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        loss = self.cal_loss(self.upload_module)

        self.upload_loss = float(loss)

        self.upload_module.model.train()
        for e in range(epochs):
            sample_idx = int(np.random.choice(
                len(self.local_training_data), 1))
            (batch_x, batch_y) = self.local_training_data[sample_idx]
            batch_x = fp.Module.change_data_device(batch_x, self.device)
            batch_y = fp.Module.change_data_device(batch_y, self.device)

            out = self.upload_module.model(batch_x)
            loss = self.criterion(out, batch_y)

            self.upload_module.model.zero_grad()
            loss.backward()
            optimizer.step()

    def train(self, epochs, lr):
        self.upload_module = copy.deepcopy(self.module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        loss = self.cal_loss(self.upload_module)

        self.upload_loss = float(loss)

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

    def create_metric_dict(self):
        metric_dict = {'test_loss': 0}
        for metric in self.metric_list:
            metric_dict[metric.name] = 0
        return metric_dict

    def test(self, test_module):

        test_module.model.eval()
        criterion = self.train_setting['criterion'].to(self.device)

        self.metric_history['training_loss'].append(
            float(self.upload_loss) if self.upload_loss is not None else None)

        metric_dict = self.create_metric_dict()

        with torch.no_grad():

            self.metric_history['local_test_number'] = self.local_test_number
            for (batch_x, batch_y) in self.local_test_data:
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)

                out = test_module.model(batch_x)
                loss = criterion(out, batch_y).item()
                metric_dict['test_loss'] += float(loss) * batch_y.shape[0]
                for metric in self.metric_list:
                    metric_dict[metric.name] += metric.calc(out, batch_y)

            self.metric_history['test_loss'].append(
                metric_dict['test_loss'] / self.local_test_number)
            for metric in self.metric_list:
                self.metric_history[metric.name].append(
                    metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['test_accuracy'].append(
                        100 * metric_dict['correct'] / self.local_test_number)

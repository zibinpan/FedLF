
import gfedplat as fp
import numpy as np
import argparse
import torch
import sys
torch.multiprocessing.set_sharing_strategy('file_system')


def outFunc(alg):
    loss_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        training_loss = metric_history['training_loss'][-1]
        if training_loss is None:
            continue
        loss_list.append(training_loss)
    loss_list = np.array(loss_list)

    local_acc_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        local_acc_list.append(metric_history['test_accuracy'][-1])
    local_acc_list = np.array(local_acc_list)
    p = np.ones(len(local_acc_list))
    local_acc_fairness = np.arccos(
        local_acc_list @ p / (np.linalg.norm(local_acc_list) * np.linalg.norm(p)))
    print(str(alg.params))

    stream_log = ""
    stream_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
    stream_log += 'round {}'.format(alg.current_comm_round) + \
        ' training_num {}'.format(alg.current_training_num) + '\n'
    stream_log += f'Mean Global Test loss: {format(np.mean(loss_list), ".6f")}' + \
        '\n' if len(loss_list) > 0 else ''
    stream_log += f'Gradient Conflict: {alg.conflict_log[-1]}' + '\n' if len(
        alg.conflict_log) > 0 else ''
    stream_log += f'Layer-wise Gradient Conflict: {alg.layer_conflict_log[-1]}' + '\n' if len(
        alg.layer_conflict_log) > 0 else ''
    stream_log += f'Direction Conflict: {alg.descent_log[-1]}' + '\n' if len(
        alg.descent_log) > 0 else ''
    stream_log += f'Layer-wise Direction Conflict: {alg.layer_descent_log[-1]}' + '\n' if len(
        alg.layer_descent_log) > 0 else ''
    stream_log += 'global model test: \n'
    stream_log += f'Local Test Acc: {format(np.mean(local_acc_list/100), ".3f")}({format(np.std(local_acc_list/100), ".3f")}), ave: {format(np.mean(local_acc_list), ".6f")}, std: {format(np.std(local_acc_list), ".6f")}, angle: {format(local_acc_fairness, ".6f")}, min: {format(np.min(local_acc_list), ".6f")}, max: {format(np.max(local_acc_list), ".6f")}' + '\n'
    stream_log += f'communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n'
    stream_log += '\n'
    alg.stream_log = stream_log + alg.stream_log
    print(stream_log)


def read_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='seed', type=int, default=1)

    parser.add_argument(
        '--device', help='device: -1, 0, 1, or ...', type=int, default=0)

    parser.add_argument('--module', help='module name;',
                        type=str, default='CNN_CIFAR10_FedFV')

    parser.add_argument('--algorithm', help='algorithm name;',
                        type=str, default='FedAvg')

    parser.add_argument('--dataloader', help='dataloader name;',
                        type=str, default='DataLoader_cifar10_non_iid')

    parser.add_argument('--SN', help='split num', type=int, default=200)

    parser.add_argument('--PN', help='pick num', type=int, default=2)

    parser.add_argument('--B', help='batch size', type=int, default=50)

    parser.add_argument('--NC', help='client_class_num', type=int, default=2)

    parser.add_argument(
        '--balance', help='balance or not for pathological separation', type=str, default='True')

    parser.add_argument(
        '--Diralpha', help='alpha parameter for dirichlet', type=float, default=0.1)

    parser.add_argument('--types', help='dataloader label types;',
                        type=str, default='default_type')

    parser.add_argument('--N', help='client num', type=int, default=100)

    parser.add_argument(
        '--C', help='select client proportion', type=float, default=1.0)

    parser.add_argument('--R', help='communication round',
                        type=int, default=3000)

    parser.add_argument('--E', help='local epochs', type=int, default=1)

    parser.add_argument('--test_interval',
                        help='test interval', type=int, default=1)

    parser.add_argument('--test_conflicts',
                        help='test conflicts', type=str, default='False')

    parser.add_argument('--sgd_step', help='sgd training',
                        type=str, default='False')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('--decay', help='learning rate decay',
                        type=float, default=0.999)
    parser.add_argument('--momentum', help='momentum', type=float, default=0.0)

    parser.add_argument('--alpha', help='alpha of FedFV',
                        type=float, default=0.1)
    parser.add_argument(
        '--tau', help='parameter tau in FedFV', type=int, default=1)

    parser.add_argument('--beta', help='beta of FedFa',
                        type=float, default=0.5)
    parser.add_argument(
        '--gamma', help='parameter gamma in FedFa', type=float, default=0.9)

    parser.add_argument(
        '--lam', help='parameter tau in Ditto', type=float, default=0.1)

    parser.add_argument(
        '--epsilon', help='parameter epsilon in FedMGDA+', type=float, default=0.1)

    parser.add_argument('--q', help='parameter q in qFedAvg',
                        type=float, default=0.1)

    parser.add_argument(
        '--mu', help='parameter mu in FedProx', type=float, default=0.0)

    parser.add_argument('--dishonest_num',
                        help='dishonest number', type=int, default=0)
    parser.add_argument('--scaled_update',
                        help='scaled update attack', type=str, default='None')
    parser.add_argument('--random_update',
                        help='random update attack', type=str, default='None')
    parser.add_argument(
        '--zero_update', help='zero update attack', type=str, default='None')

    try:
        parsed = vars(parser.parse_args())
        return parsed
    except IOError as msg:
        parser.error(str(msg))


def initialize(params):
    fp.setup_seed(seed=params['seed'])
    device = torch.device(
        'cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
    Module = getattr(sys.modules['gfedplat'], params['module'])
    module = Module(device)
    Dataloader = getattr(sys.modules['gfedplat'], params['dataloader'])
    data_loader = Dataloader(
        params=params, input_require_shape=module.input_require_shape)
    module.generate_model(data_loader.input_data_shape,
                          data_loader.target_class_num)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, module.model.parameters(
    )), lr=params['lr'], momentum=params['momentum'])
    train_setting = {'criterion': torch.nn.CrossEntropyLoss(
    ), 'optimizer': optimizer, 'lr_decay': params['decay'], 'sgd_step': eval(params['sgd_step'])}
    test_interval = params['test_interval']
    dishonest_num = params['dishonest_num']
    scaled_update = eval(params['scaled_update'])
    if scaled_update is not None:
        scaled_update = float(scaled_update)
    dishonest = {'dishonest_num': dishonest_num,
                 'scaled_update': scaled_update,
                 'random_update': eval(params['random_update']),
                 'zero_update': eval(params['zero_update'])}
    test_conflicts = eval(params['test_conflicts'])
    Algorithm = getattr(sys.modules['gfedplat'], params['algorithm'])
    algorithm = Algorithm(data_loader=data_loader,
                          module=module,
                          device=device,
                          train_setting=train_setting,
                          client_num=data_loader.pool_size,
                          online_client_num=int(
                              data_loader.pool_size * params['C']),
                          metric_list=[fp.Correct()],
                          max_comm_round=params['R'],
                          max_training_num=None,
                          epochs=params['E'],
                          outFunc=outFunc,
                          write_log=True,
                          dishonest=dishonest,
                          test_conflicts=test_conflicts,
                          params=params,)
    algorithm.test_interval = test_interval
    return data_loader, algorithm


if __name__ == '__main__':
    params = read_params()
    data_loader, algorithm = initialize(params)
    algorithm.run()

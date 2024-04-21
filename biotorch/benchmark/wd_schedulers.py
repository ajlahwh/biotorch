import torch

class MultiStepWD:
    def __init__(self, optimizer, milestones, gamma=0.1, verbose=True):
        self.optimizer = optimizer
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.verbose = verbose
        self.current_epoch = 0

    def step(self):
        for milestone in self.milestones:
            if self.current_epoch == milestone:
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] *= self.gamma
                if self.verbose:
                    print(f'Epoch {self.current_epoch}: updated weight decay to {param_group["weight_decay"]}')
        self.current_epoch += 1

def create_wd_scheduler(wd_scheduler_config, optimizer):
    if wd_scheduler_config is None:
        wd_scheduler = MultiStepWD(
            optimizer,
            milestones=[],
            gamma=1.0,
            verbose=True)
    elif wd_scheduler_config['type'] == 'multistep_wd':
        gamma = wd_scheduler_config['gamma']
        wd_scheduler = MultiStepWD(
            optimizer,
            milestones=wd_scheduler_config['milestones'],
            gamma=gamma,
            verbose=True)
    else:
        raise ValueError('Optimizer type {} not supported'.format(wd_scheduler_config['type']))

    return wd_scheduler

from torch.optim.lr_scheduler import _LRScheduler

class LinearCyclicalLR(_LRScheduler):
        def __init__(self, optimizer, T_down, T_up , eta_min=0, last_epoch=-1):
            self.T_sum = T_down + T_up
            self.T_down = T_down
            self.T_up = T_up
            self.eta_min = eta_min
            super(LinearCyclicalLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            last_step = self.last_epoch % self.T_sum
            return [-(base_lr-self.eta_min)*last_step / self.T_down + base_lr if last_step < self.T_down
                    else (base_lr-self.eta_min)*last_step/ self.T_up + (self.T_sum/ self.T_up)*self.eta_min - (self.T_down/self.T_up)*base_lr
                    for base_lr in self.base_lrs]


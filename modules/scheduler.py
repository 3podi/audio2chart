import math
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing
    When used with interval='step', warmup_steps and max_steps are in training steps
    """
    
    def __init__(self, optimizer, warmup_steps, max_steps, warmup_start_lr=1e-8, eta_min=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps  # Number of warmup STEPS
        self.max_steps = max_steps        # Total training STEPS  
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup (self.last_epoch is actually step count when interval='step')
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_steps
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

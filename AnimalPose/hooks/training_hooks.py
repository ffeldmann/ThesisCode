from edflow import get_logger
from edflow.hooks.hook import Hook


class AdjustLearningRate(Hook):
    def __init__(self, config, optimizer):
        self.logger = get_logger(self)
        self.optimizer = optimizer
        self.config = config

    def before_epoch(self, epoch):
        if epoch in range(3, 5):
            """
            Sets the learning rate to the initial LR decayed by 10 within the first 3-5 epochs
            """
            lr = self.config["lr"] * (0.1 ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.logger.info(f"Decreased learning rate to {lr}.")

        if epoch > 5:
            lr = self.config["lr"] * (0.1 ** (epoch // 10))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.logger.info(f"Decreased learning rate to {lr}.")
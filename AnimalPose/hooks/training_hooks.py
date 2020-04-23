from edflow import get_logger
from edflow.hooks.hook import Hook


class AdjustLearningRate(Hook):
    def __init__(self, config, scheduler, losses):
        self.scheduler = get_logger(self)
        self.optimizer = scheduler
        self.losses = losses

    def before_epoch(self, epoch):
        self.scheduler.step(self.losses["batch"]["total"], epoch=self.get_epoch_step())
        # if epoch in range(3, 5):
        #     """
        #     Sets the learning rate to the initial LR decayed by 10 within the first 3-5 epochs
        #     """
        #     lr = self.config["lr"] * (0.1 ** epoch)
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     self.logger.info(f"Decreased learning rate to {lr}.")
        #
        # if epoch > 5:
        #     lr = self.config["lr"] * (0.1 ** (epoch // 10))
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     self.logger.info(f"Decreased learning rate to {lr}.")


class KLDecay(Hook):
    def __init__(self, config):
        self.logger = get_logger(self)

    def before_epoch(self, epoch):

        self.scheduler.step(self.losses["batch"]["total"], epoch=self.get_epoch_step())
        # if epoch in range(3, 5):
        #     """
        #     Sets the learning rate to the initial LR decayed by 10 within the first 3-5 epochs
        #     """
        #     lr = self.config["lr"] * (0.1 ** epoch)
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     self.logger.info(f"Decreased learning rate to {lr}.")
        #
        # if epoch > 5:
        #     lr = self.config["lr"] * (0.1 ** (epoch // 10))
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     self.logger.info(f"Decreased learning rate to {lr}.")
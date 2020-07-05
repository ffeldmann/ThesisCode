import torch
import torch.nn as nn
from edflow.util import retrieve
from AnimalPose.utils.perceptual_loss.models import PerceptualLoss

"""
mu ist hyperparameter wie stark der loss angepasst wird,
eps ist der maximal zulässige reconstruction loss 
(hier l2 loss, kann natürlich durch perceptual loss ausgetauscht werden).
train op sollte nach jedem train schritt ausgeführt werden um lambda zu updaten
"""


class LossConstrained(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_init = retrieve(config, "LossConstrained/lambda_init", default=100000.0)
        self.register_buffer("lambda_", torch.ones(size=()) * self.lambda_init)
        self.mu = retrieve(config, "LossConstrained/mu", default=0.01)  # anpassung loss
        self.eps = retrieve(config, "LossConstrained/eps", default=0.4)  # max rec loss # 30.72)  # mean of 0.01
        net = self.config["losses"]["perceptual_network"]
        self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=True, spatial=False).to("cuda")
        self.device = "cuda"

    def forward(self, inputs, reconstructions, mu, logvar, global_step):
        # L2 Loss
        # [B, C, W, H] - [B, C, W, H]
        # rec_loss = (inputs - reconstructions) ** 2
        # rec_sum = torch.sum(rec_loss) / rec_loss.shape[0]
        # rec_mean = rec_loss.mean()
        # Perceptual
        rec_loss = self.perceptual_loss(torch.from_numpy(inputs).float().to(self.device),
                                        reconstructions.to(self.device), True)
        rec_sum = torch.sum(rec_loss) / rec_loss.shape[0]
        rec_mean = rec_loss.mean()

        gain = rec_sum - self.eps
        active = (self.mu * gain >= -self.lambda_).detach().type(torch.float)

        nll_loss = active * (self.lambda_.cuda() * gain + self.mu / 2.0 * gain ** 2)

        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1. - logvar)

        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        def train_op():
            new_lambda = self.lambda_ + self.mu * gain
            new_lambda = torch.clamp(new_lambda, min=0.0)
            self.lambda_.data.copy_(new_lambda)
            if self.lambda_ <= 10:
                self.mu = self.mu // 2
            if self.lambda_ <= 2:
                self.mu = self.mu // 2

        loss = nll_loss.cuda() + kl_loss.cuda()
        log = {"images": {"inputs": inputs, "reconstructions": reconstructions},
               "scalars": {"loss": loss,
                           "lambda_": self.lambda_,
                           "gain": gain,
                           "active": active,
                           "kl_loss": kl_loss,
                           "nll_loss": nll_loss,
                           "rec_loss": rec_mean,
                           "mu": self.mu,
                           "eps": self.eps,
                           }}
        return loss.to(self.device), log, train_op

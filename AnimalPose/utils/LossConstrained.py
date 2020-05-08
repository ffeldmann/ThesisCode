import torch
import torch.nn as nn
from edflow.util import retrieve

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
        self.lambda_init = retrieve(config, "Loss/lambda_init", default=1.0)
        self.register_buffer("lambda_", torch.ones(size=()) * self.lambda_init)
        self.mu = retrieve(config, "Loss/mu", default=0.05)
        self.eps = retrieve(config, "Loss/eps", default=30.72)  # mean of 0.01

    def forward(self, inputs, reconstructions, samples, posteriors, global_step):
        rec_loss = (inputs - reconstructions) ** 2
        rec_sum = torch.sum(rec_loss) / rec_loss.shape[0]
        rec_mean = rec_loss.mean()

        gain = rec_sum - self.eps
        active = (self.mu * gain >= -self.lambda_).detach().type(torch.float)

        nll_loss = active * (
                self.lambda_ * gain +
                self.mu / 2.0 * gain ** 2)

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        def train_op():
            new_lambda = self.lambda_ + self.mu * gain
            new_lambda = torch.clamp(new_lambda, min=0.0)
            self.lambda_.data.copy_(new_lambda)

        loss = nll_loss + kl_loss
        log = {"images": {"inputs": inputs, "reconstructions": reconstructions},
               "scalars": {"loss": loss,
                           "lambda_": self.lambda_,
                           "gain": gain,
                           "active": active,
                           "kl_loss": kl_loss,
                           "nll_loss": nll_loss,
                           "rec_loss": rec_mean, }}
        return loss, log, train_op

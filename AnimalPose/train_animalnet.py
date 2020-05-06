import time

import torch
import torch.nn.functional
import torch.optim as optim
from edflow import TemplateIterator
from edflow.util import retrieve

from AnimalPose.utils.loss_utils import MSELossInstances, VGGLossWithL1
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy
from AnimalPose.utils.perceptual_loss.models import PerceptualLoss
import numpy as np


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.device = "cuda" if self.cuda else "cpu"
        self.variational = self.config["variational"]["active"]
        self.encoder_2 = True if self.config["encoder_2"] else False
        self.kl_weight = self.config["variational"]["kl_weight"]

        self.start_step, self.stop_step, self.start_weight, self.stop_weight = self.config["variational"]["start_step"], \
                                                                               self.config["num_steps"], self.kl_weight, \
                                                                               self.config["variational"]["stop_weight"]
        # vgg loss
        if self.config["losses"]["vgg"]:
            self.vggL1 = VGGLossWithL1(gpu_ids=[0],
                                       l1_alpha=self.config["losses"]["vgg_l1_alpha"],
                                       vgg_alpha=self.config["losses"]["vgg_alpha"]).to(self.device)
        if self.config["losses"]["KL"]:
            self.klloss = torch.nn.KLDivLoss(reduction="batchmean")
        # initalize perceptual loss if possible
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=False).to(
                self.device)
        if self.cuda:
            self.model.cuda()
        # hooks
        # if "pretrained_checkpoint" in self.config.keys():
        #     self.hooks.append(
        #         RestorePretrainedSDCHook(
        #             pretrained_checkpoint=self.config["pretrained_checkpoint"],
        #             model=self.model,
        #         )
        #     )

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def criterion(self, targets, predictions, mu=None, logvar=None):
        # calculate losses
        crit = torch.nn.MSELoss()
        batch_losses = {}
        if self.config["losses"]["L2"]:
            batch_losses["L2_loss"] = crit(torch.from_numpy(targets), predictions.cpu()).to(self.device)
        if self.variational:
            # assert self.config["losses"]["L2"], "L2 loss necessary here!"
            KLD = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1. - logvar)
            batch_losses["KL"] = KLD * self.kl_weight

        if self.config["losses"]["perceptual"]:
            batch_losses["perceptual"] = torch.mean(
                self.perceptual_loss(torch.from_numpy(targets).float().to(self.device),
                                     predictions.to(self.device), True))
        if self.config["losses"]["vgg"]:
            batch_losses["vgg"] = self.vggL1(torch.from_numpy(targets).float().to(self.device),
                                             predictions.to(self.device))
        batch_losses["total"] = sum(
            [
                batch_losses[key]
                for key in batch_losses.keys()
            ]
        )

        return batch_losses

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)
        if self.get_global_step() >= self.start_step and is_train:
            if self.variational:
                # self.logger.info(f"Global step: {self.get_global_step()}")
                self.logger.info(f"Global step: {self.get_global_step()}")
                prev = self.kl_weight
                if self.config["variational"]["decay"]:
                    self.kl_weight = self.kl_weight * 0.99
                    self.logger.info(f"Decay prev kl_weight {prev} to {self.kl_weight}.")
                else:  # start_step, stop_step, start_weight, stop_weight
                    relative_global_step = self.get_global_step() - self.start_step
                    self.kl_weight = self.start_weight + ((self.stop_weight - self.start_weight) * (
                            relative_global_step / (self.stop_step - self.start_step)))  # * 1.001
                    self.logger.info(f"Increase prev kl_weight {prev} to {self.kl_weight}.")

        # (batch_size, width, height, channel)
        inputs0 = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2)).to("cuda")
        if self.encoder_2:
            inputs1 = numpy2torch(kwargs["inp1"].transpose(0, 3, 1, 2)).to("cuda")
        # inputs now
        # (batch_size, channel, width, height)
        # compute model

        if self.encoder_2:
            if self.variational:
                predictions, mu, logvar = model(inputs0, inputs1)
                # Do only in test
                if not is_train:
                    inputs1_flipped = torch.flip(inputs1, [0])  # flip the tensor in zero dimension
                    kl_test_preds, _, _ = model(inputs0, inputs1_flipped)
            else:
                predictions = model(inputs0, inputs1)
        else:
            if self.variational:
                predictions, mu, logvar = model(inputs0)
            else:
                predictions = model(inputs0)

        # compute loss
        # Target heatmaps, predicted heatmaps, gt_coords
        if self.variational:
            if self.encoder_2:
                losses = self.criterion(kwargs["inp0"].transpose(0, 3, 1, 2), predictions, mu, logvar)
            else:
                losses = self.criterion(kwargs["inp0"].transpose(0, 3, 1, 2), predictions, mu, logvar)
        else:
            losses = self.criterion(kwargs["inp0"].transpose(0, 3, 1, 2), predictions)

        def train_op():
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

        def log_op():
            from edflow.data.util import adjust_support
            is_train = self.get_split() == "train"
            logs = {
                "images": {
                    "image_input_0": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "outputs": adjust_support(torch2numpy(predictions).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                },
                "scalars": {
                    "loss": losses["total"],
                },
            }

            if self.encoder_2:
                logs["images"]["image_input_1"] = adjust_support(torch2numpy(inputs1).transpose(0, 2, 3, 1), "-1->1",
                                                                 "0->1")
                if not is_train and self.variational:
                    logs["images"]["image_input_1_flipped"] = adjust_support(
                        torch2numpy(inputs1_flipped).transpose(0, 2, 3, 1), "-1->1", "0->1")
                    logs["images"]["pose_reconstruction"] = adjust_support(
                        torch2numpy(kl_test_preds).transpose(0, 2, 3, 1), "-1->1", "0->1")
            if self.kl_weight:
                logs["scalars"]["kl_weight"] = self.kl_weight
            if self.config["losses"]["L2"]:
                logs["scalars"]["L2_loss"] = losses["L2_loss"]
            if self.config["losses"]["perceptual"]:
                logs["scalars"]["perceptual"] = losses["perceptual"]
            if self.config["losses"]["KL"] and self.variational:
                logs["scalars"]["KL"] = losses["KL"]
            if self.config["losses"]["vgg"]:
                logs["scalars"]["vgg"] = losses["vgg"]
            return logs

        def eval_op():
            # percentage correct keypoints pck
            # return {
            # "outputs": np.array(predictions.cpu().detach().numpy()),
            # TODO in which shape is the outputs necessary for evaluation?
            # "labels": {k: [v.cpu().detach().numpy()] for k, v in losses["batch"].items()},
            # }
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

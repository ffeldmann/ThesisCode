import time

import torch
import torch.nn.functional
import torch.optim as optim
from edflow import TemplateIterator
from edflow.util import retrieve

from AnimalPose.data.util import heatmap_to_image, make_stickanimal
from AnimalPose.hooks.model import RestorePretrainedSDCHook
from AnimalPose.utils.loss_utils import heatmap_loss, keypoint_loss
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.encoder2 = True if self.config["encoder_2"] else False
        if self.cuda:
            self.model.cuda()
        # hooks
        if "pretrained_checkpoint" in self.config.keys():
            self.hooks.append(
                RestorePretrainedSDCHook(
                    pretrained_checkpoint=self.config["pretrained_checkpoint"],
                    model=self.model,
                )
            )

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

    def criterion(self, targets, predictions):
        # calculate losses
        crit = torch.nn.MSELoss()

        instance_losses = {}
        if self.config["losses"]["L2"]:
            instance_losses["L2_loss"] = crit(torch.from_numpy(targets), predictions)
        instance_losses["total"] = sum(
            [
                instance_losses[key]
                for key in instance_losses.keys()
            ]
        )

        # reduce to batch granularity
        batch_losses = {k: v.mean() for k, v in instance_losses.items()}
        losses = dict(instances=instance_losses, batch=batch_losses)
        return losses

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)

        # TODO need (batch_size, channel, width, height)
        # (batch_size, width, height, channel)
        inputs0 = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2)).to("cuda")
        inputs1 = numpy2torch(kwargs["inp1"].transpose(0, 3, 1, 2)).to("cuda")
        # inputs now
        # (batch_size, channel, width, height)
        # compute model
        if self.encoder2:
            predictions = model(inputs0, inputs1)
        else:
            predictions = model(inputs0)
        # compute loss
        # Target heatmaps, predicted heatmaps, gt_coords
        losses = self.criterion(kwargs["inp0"].transpose(0,3,1,2), predictions.cpu())

        def train_op():
            before = time.time()
            self.optimizer.zero_grad()
            losses["batch"]["total"].backward()
            self.optimizer.step()
            if retrieve(self.config, "debug_timing", default=False):
                self.logger.info("train step needed {} s".format(time.time() - before))

        def log_op():
            from edflow.data.util import adjust_support
            logs = {
                "images": {
                    "image_input_0": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1"),
                    "image_input_1": adjust_support(torch2numpy(inputs1).transpose(0, 2, 3, 1), "-1->1"),
                    "outputs": adjust_support(torch2numpy(predictions).transpose(0, 2, 3, 1),"-1->1"),
                },
                "scalars": {
                    "loss": losses["batch"]["total"],
                },
            }
            if self.config["losses"]["L2"]:
                logs["scalars"]["L2_loss"] = losses["batch"]["L2_loss"]
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
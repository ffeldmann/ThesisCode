import time

import torch
import torch.nn.functional
import torch.optim as optim
from edflow import TemplateIterator

from AnimalPose.data.util import make_stickanimal
from AnimalPose.utils.image_utils import heatmaps_to_coords, get_color_heatmaps
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy
from AnimalPose.utils.log_utils import plot_input_target_keypoints
from AnimalPose.utils.loss_utils import percentage_correct_keypoints
from edflow.data.util import adjust_support
from AnimalPose.hooks.training_hooks import AdjustLearningRate
from AnimalPose.utils.loss_utils import MSELossInstances, L1LossInstances
from AnimalPose.utils.tensor_utils import sure_to_torch, sure_to_numpy


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Initialize Loss functions
        self.mse_instance = MSELossInstances()
        self.l1_instance = L1LossInstances()
        # self.normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        if self.cuda:
            self.model.cuda()
        # hooks
        if self.config["adjust_learning_rate"]:
            self.hooks.append(
                AdjustLearningRate(self.config, self.optimizer)
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

    def criterion(self, targets, predictions, gt_coords):
        # make sure everything is a torch tensor
        targets = sure_to_torch(targets)
        predictions = sure_to_torch(predictions)

        # calculate losses
        instance_losses = {}
        if self.config["losses"]["L2"]:
            instance_losses["L2"] = self.mse_instance(targets, predictions)
        if self.config["losses"]["L1"]:
            instance_losses["L1"] = self.l1_instance(targets, predictions)

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

        # kwargs["inp"]
        # (batch_size, width, height, channel)
        inputs = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2)).to("cuda")
        # inputs now (batch_size, channel, width, height)
        # compute model
        predictions = model(inputs)
        # compute loss
        # Target heatmaps, predicted heatmaps, gt_coords
        losses = self.criterion(kwargs["targets"], predictions.cpu(), kwargs["kps"])

        def train_op():
            self.optimizer.zero_grad()
            losses["batch"]["total"].backward()
            self.optimizer.step()

        def log_op():

            PCK_THRESH = [0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5]
            if self.config['pck_alpha'] not in PCK_THRESH: PCK_THRESH.append(self.config["pck_alpha"])

            coords = heatmaps_to_coords(predictions, thresh=self.config["hm"]["thresh"])
            pck = {t: percentage_correct_keypoints(kwargs["kps"], coords, t, self.config["pck"]["type"]) for t in
                   PCK_THRESH}
            logs = {
                "images": {
                    # Image input not needed, because stickanimal is printed on input image
                    #"image_input": adjust_support(torch2numpy(inputs).transpose(0, 2, 3, 1), "-1->1"),
                    "outputs": adjust_support(get_color_heatmaps(predictions), "-1->1").transpose(0, 2, 3, 1),
                    "targets": adjust_support(get_color_heatmaps(kwargs["targets"]), "-1->1").transpose(0, 2, 3, 1),
                    "inputs_with_stick": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), kwargs["kps"]),
                    "stickanimal": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1),
                                                    torch2numpy(predictions)),
                },
                "scalars": {
                    "loss": losses["batch"]["total"],
                    f"PCK@{self.config['pck_alpha']}": pck[self.config['pck_alpha']][0],
                },
                "figures": {
                    "Keypoint Mapping": plot_input_target_keypoints(torch2numpy(inputs).transpose(0, 2, 3, 1),
                                                                    # get BHWC
                                                                    torch2numpy(predictions),  # stay BCHW
                                                                    kwargs["kps"]),
                }
            }
            if self.config["losses"]["L2"]:
                logs["scalars"]["L2"] = losses["batch"]["L2"]
            if self.config["losses"]["L1"]:
                logs["scalars"]["L1"] = losses["batch"]["L1"]

            # Add left and right
            def accumulate_side(index, value, side="L"):
                """
                Accumulates PCK parts from the left or right side of the animal.
                Args:
                    index:
                    value:
                    side:

                Returns:

                """
                if side in self.dataset.get_idx_parts(index):
                    try:
                        logs["scalars"][f"PCK@{self.config['pck_alpha']}_{side}side"] += value
                    except:
                        logs["scalars"][f"PCK@{self.config['pck_alpha']}_{side}side"] = value

            if self.config["pck"]["pck_multi"]:
                for key, val in pck.items():
                    # get mean value for pck at given threshold
                    logs["scalars"][f"PCK@_{key}"] = val[0]
                for idx, part in enumerate(val[1]):
                    logs["scalars"][f"PCK@_{key}_{self.dataset.get_idx_parts(idx)}"] = part
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

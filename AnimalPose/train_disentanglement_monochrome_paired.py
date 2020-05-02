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
        self.mseloss = torch.nn.MSELoss()
        self.L1loss = torch.nn.L1Loss()
        # vgg loss
        if self.config["losses"]["vgg"]:
            self.vggL1 = VGGLossWithL1(gpu_ids=[0],
                                       l1_alpha=self.config["losses"]["vgg_l1_alpha"],
                                       vgg_alpha=self.config["losses"]["vgg_alpha"]).to(self.device)

        # initalize perceptual loss if possible
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=True).to(
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
        batch_losses = {}

        if self.config["losses"]["L1"]:
            batch_losses["L1"] = self.mseloss(targets, predictions)
        if self.config["losses"]["L2"]:
            batch_losses["L2"] = self.mseloss(targets, predictions)
        if self.config["losses"]["perceptual"]:
            batch_losses["perceptual"] = torch.mean(
                self.perceptual_loss(targets, predictions, True))
        if self.config["losses"]["vgg"]:
            batch_losses["vgg"] = self.vggL1(targets, predictions.to(self.device))
        batch_losses["total"] = sum(
            [
                batch_losses[key]
                for key in batch_losses.keys()
            ]
        )

        return batch_losses["total"]

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)

        # (batch_size, width, height, channel)
        # We half the batch, first half for pose and second for appearance
        inputs0 = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2)).to("cuda")
        inputs1 = numpy2torch(kwargs["inp1"].transpose(0, 3, 1, 2)).to("cuda")
        # inputs0, inputs1 = torch.split(inputs0, int(inputs0.size(0) / 2), dim=0)

        # Autoencoder (a)
        pred0, appearance0, pose0 = model(inputs0)  # appearance
        pred1, appearance1, pose1 = model(inputs1)  # pose

        # Mixed Reconstruction
        mixed_reconstruction = model(enc_appearance=appearance0, enc_pose=pose1, mixed_reconstruction=True)
        # cycle consistency
        # appearance_recon, pose_recon, latent_appearance, latent_pose = model(mixed_reconstruction,
        #                                                                     enc_appearance=appearance0, enc_pose=pose1,

        loss_appearance = self.criterion(inputs0, pred0) * 1
        loss_pose = self.criterion(inputs1, pred1) * 1
        loss_mixed_reconstruction = self.criterion(inputs1, mixed_reconstruction)
        loss = loss_appearance + loss_pose + loss_mixed_reconstruction
        # loss_frank = self.criterion(mixed_reconstruction, pred1)

        flip_test, _, _ = model(input=inputs0, input2=torch.flip(inputs0, [0]))

        def train_op():
            self.optimizer.zero_grad()

            # Freeze pose encoder
            for param in model.backbone.parameters():
                param.requires_grad = False
            loss_appearance.backward(retain_graph=True)
            self.optimizer.step()
            # # release pose encoder weights
            for param in model.backbone.parameters():
                param.requires_grad = True
            for param in model.backbone2.parameters():
                param.requires_grad = False
            loss_mixed_reconstruction.backward(retain_graph=True)
            loss_pose.backward()
            self.optimizer.step()
            for param in model.backbone2.parameters():
                param.requires_grad = True

        def log_op():
            from edflow.data.util import adjust_support
            logs = {
                "images": {
                    "inputs0": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "inputs1": adjust_support(torch2numpy(inputs1).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "pred0": adjust_support(torch2numpy(pred0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "pred1": adjust_support(torch2numpy(pred1).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "mixed_reconstruction": adjust_support(torch2numpy(mixed_reconstruction).transpose(0, 2, 3, 1),
                                                           "-1->1", "0->1"),
                    "input0_flipped": adjust_support(torch2numpy(torch.flip(inputs0, [0])).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "flip_test": adjust_support(torch2numpy(flip_test).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                },
                "scalars": {
                    "loss": loss_appearance + loss_pose + loss_mixed_reconstruction,
                    "loss_appearance": loss_appearance,
                    "loss_pose": loss_pose,
                    "loss_mixed_reconstruction": loss_mixed_reconstruction,
                },
            }
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

import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import torchvision
from edflow import TemplateIterator

from AnimalPose.data.util import make_stickanimal
from AnimalPose.utils.image_utils import heatmaps_to_coords, heatmaps_to_image
from AnimalPose.utils.log_utils import plot_input_target_keypoints
from AnimalPose.utils.loss_utils import VGGLossWithL1
from AnimalPose.utils.loss_utils import percentage_correct_keypoints
from AnimalPose.utils.perceptual_loss.models import PerceptualLoss
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy
from AnimalPose.utils.tensor_utils import sure_to_torch, sure_to_numpy


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.device = "cuda" if self.cuda else "cpu"
        self.mseloss = torch.nn.MSELoss()
        self.L1loss = torch.nn.L1Loss()
        self.sigmoid = torch.nn.Sigmoid()
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
        inputs = numpy2torch(kwargs["inp1"].transpose(0, 3, 1, 2)).to("cuda")
        #inputs_heatmaps = numpy2torch(kwargs["targets0"]).to("cuda")
        inputs0, inputs1 = torch.split(inputs, int(inputs.size(0) / 2), dim=0)
        #inputs_heatmaps0, inputs_heatmaps1 = torch.split(inputs_heatmaps, int(inputs_heatmaps.size(0) / 2), dim=0)

        # Autoencoder (a)
        pred0, appearance0, pose0 = model(inputs0)  # appearance
        pred1, appearance1, pose1 = model(inputs1)  # pose
        pred0 = self.sigmoid(pred0)
        pred1 = self.sigmoid(pred1)

        autoencoder_loss = 0
        autoencoder_loss += self.criterion(inputs0, pred0) * 1  # Eq2
        autoencoder_loss += self.criterion(inputs1, pred1) * 1  # Eq2

        # (b) Mixed Reconstruction
        mixed_reconstruction = model(enc_appearance=appearance0, enc_pose=pose1, mixed_reconstruction=True)
        mixed_reconstruction = self.sigmoid(mixed_reconstruction)
        # (c) cycle consistency
        #model.eval()
        pose_recon, appearance_recon, latent_appearance_hat, latent_pose_hat = model(mixed_reconstruction,
                                                                                     enc_appearance=appearance1,
                                                                                     enc_pose=pose0,
                                                                                     cycle=True)
        pose_recon = self.sigmoid(pose_recon)
        appearance_recon = self.sigmoid(appearance_recon)
        #model.train()

        cycle_loss = 0
        cycle_loss += self.criterion(appearance_recon, inputs0) * 1  # Eq4
        cycle_loss += self.criterion(pose_recon, inputs1) * 1  # Eq4

        #latent_loss = 0
        #latent_loss += self.L1loss(latent_pose_hat, pose1)  # Eq 5
        #latent_loss += self.L1loss(latent_appearance_hat, appearance0)  # Eq5

        #hm0 = model(inputs0, heatmap=True)  # Heatmaps
        #hm1 = model(inputs1, heatmap=True)

        #pose_predictions = torch.cat((hm0, hm1), dim=0)
        #pose_loss = 0
        #pose_loss += self.criterion(hm0, inputs_heatmaps0)  # Eq1
        #pose_loss += self.criterion(hm1, inputs_heatmaps1)  # Eq1

        def train_op():
            self.optimizer.zero_grad()

            autoencoder_loss.backward(retain_graph=True)
            cycle_loss.backward(retain_graph=True)
            # Freeze pose encoder
            #for param in model.backbone.parameters():
            #    param.requires_grad = False
            self.optimizer.step()
            #self.optimizer.zero_grad()
            # Release pose encoder
            #for param in model.backbone.parameters():
            #    param.requires_grad = True
            #pose_loss.backward()
            #self.optimizer.step()

            # freeze decoder
            # for param in model.head.parameters():
            #    param.requires_grad = False
            # for param in model.backbone2.parameters():
            #    param.requires_grad = False
            # cycle_loss.backward()
            # # release pose encoder weights

        def log_op():
            from edflow.data.util import adjust_support

            #PCK_THRESH = [0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5]
            #if self.config['pck_alpha'] not in PCK_THRESH: PCK_THRESH.append(self.config["pck_alpha"])

            #coords, _ = heatmaps_to_coords(pose_predictions.clone(), thresh=self.config["hm"]["thresh"])
            #pck = {t: percentage_correct_keypoints(kwargs["kps"], coords, t, self.config["pck"]["type"]) for t in
            #       PCK_THRESH}

            logs = {
                "images": {
                    "inputs0": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "inputs1": adjust_support(torch2numpy(inputs1).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "pred0": adjust_support(torch2numpy(pred0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "pred1": adjust_support(torch2numpy(pred1).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "mixed_reconstruction": adjust_support(torch2numpy(mixed_reconstruction).transpose(0, 2, 3, 1),
                                                           "-1->1", "0->1"),
                    "cycle0": adjust_support(torch2numpy(pose_recon).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "cycle1": adjust_support(torch2numpy(appearance_recon).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    #"targets": adjust_support(heatmaps_to_image(kwargs["targets0"]).transpose(0, 2, 3, 1), "-1->1"),
                    #"inputs_with_stick": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), kwargs["kps0"]),
                    #"stickanimal": make_stickanimal(torch2numpy(inputs).transpose(0, 2, 3, 1), pose_predictions.clone(),
                    #                                thresh=self.config["hm"]["thresh"]),
                    #"pred_hm": adjust_support(heatmaps_to_image(torch2numpy(pose_predictions)).transpose(0, 2, 3, 1),
                    #                          "-1->1", "0->1"),
                },
                "scalars": {
                    "loss": autoencoder_loss + cycle_loss,# + pose_loss,
                    "loss_autoencoder": autoencoder_loss,
                    "recon_loss": cycle_loss,
                    #"pose_loss": pose_loss,
                    #f"PCK@{self.config['pck_alpha']}": np.around(pck[self.config['pck_alpha']][0], 5),
                },
            }

            # if self.config["pck"]["pck_multi"]:
            #     for key, val in pck.items():
            #         # get mean value for pck at given threshold
            #         logs["scalars"][f"PCK@{key}"] = np.around(val[0], 5)
            #         for idx, part in enumerate(val[1]):
            #             logs["scalars"][f"PCK@{key}_{self.dataset.get_idx_parts(idx)}"] = np.around(part, 5)

            # gridded_outputs = np.expand_dims(sure_to_numpy(torchvision.utils.make_grid(
            #    predictions[0], nrow=10)), 0).transpose(1, 2, 3, 0)
            # gridded_targets = np.expand_dims(sure_to_numpy(torchvision.utils.make_grid(
            #    sure_to_torch(kwargs["targets"])[0], nrow=10)), 0).transpose(1, 2, 3, 0)
            #
            # logs = {
            #     "images": {
            #         # Image input not needed, because stick animal is printed on input image
            #         # "image_input": adjust_support(torch2numpy(inputs).transpose(0, 2, 3, 1), "-1->1"),
            #         #"first_pred": adjust_support(gridded_outputs, "-1->1", "0->1"),
            #         #"first_targets": adjust_support(gridded_targets, "-1->1", "0->1"),
            #         "outputs": adjust_support(heatmaps_to_image(torch2numpy(predictions)).transpose(0, 2, 3, 1),
            #                                   "-1->1", "0->1"),

            #     },
            #     "scalars": {
            #         "loss": losses["total"],
            #         "learning_rate": self.optimizer.state_dict()["param_groups"][0]["lr"],
            #         f"PCK@{self.config['pck_alpha']}": np.around(pck[self.config['pck_alpha']][0], 5),
            #     },
            #     "figures": {
            #         "Keypoint Mapping": plot_input_target_keypoints(torch2numpy(inputs).transpose(0, 2, 3, 1),
            #                                                         # get BHWC
            #                                                         torch2numpy(predictions),  # stay BCHW
            #                                                         kwargs["kps"], coords),
            #     }
            # }
            # if self.config["losses"]["L2"]:
            #     logs["scalars"]["L2"] = losses["L2"]
            # if self.config["losses"]["L1"]:
            #     logs["scalars"]["L1"] = losses["L1"]
            # if self.config["losses"]["perceptual"]:
            #     logs["scalars"]["perceptual"] = losses["perceptual"]
            #

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

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from edflow.project_manager import ProjectManager
from edflow.hooks.hook import Hook
import time

import numpy as np
from edflow import TemplateIterator, get_logger
from edflow.util import retrieve, walk
from edflow.data.util import adjust_support
from AnimalPose.hooks.model import RestorePretrainedSDCHook, TrainHeadTailFirstNHook
from AnimalPose.utils.loss_utils import (
    update_loss_weights_inplace,
    L1LossInstances,
    MaskedL1LossInstances,
    MSELossInstances,
    aggregate_kl_loss,
    PerceptualLossInstances,
)

def np2pt(array):
    tensor = torch.tensor(array, dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def pt2np(tensor):
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (0, 2, 3, 1))
    return array



class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.L1LossInstances = L1LossInstances()
        self.MSELossInstances = MSELossInstances()
        if "masked_L1" in self.config["losses"]:
            self.MaskedL1LossInstances = MaskedL1LossInstances(
                self.config["losses"]["masked_L1"]
            )
        if "perceptual" in self.config["losses"]:
            self.PerceptualLossInstances = PerceptualLossInstances(
                self.config["losses"]["perceptual"]
            )

        if torch.cuda.is_available():
            self.model.cuda()
            self.L1LossInstances.cuda()
            self.MSELossInstances.cuda()
            if "masked_L1" in self.config["losses"]:
                self.MaskedL1LossInstances.cuda()
            if "perceptual" in self.config["losses"]:
                self.PerceptualLossInstances.cuda()

        # hooks
        if "pretrained_checkpoint" in self.config.keys():
            self.hooks.append(
                RestorePretrainedSDCHook(
                    pretrained_checkpoint=self.config["pretrained_checkpoint"],
                    model=self.model,
                )
            )
        if "train_head_tail_first" in self.config.keys():
            self.hooks.append(
                TrainHeadTailFirstNHook(
                    model=self.model,
                    n=self.config["train_head_tail_first"]["n"],
                    layers_to_train=self.config["train_head_tail_first"][
                        "layers_to_train"
                    ],
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

    def criterion(self, inputs, predictions):
        # update kl weights
        update_loss_weights_inplace(self.config["losses"], self.get_global_step())

        # calculate losses
        instance_losses = {}

        if "color_L1" in self.config["losses"].keys():
            instance_losses["color_L1"] = self.L1LossInstances(
                predictions["image"], inputs["target"]
            )

        if "masked_L1" in self.config["losses"].keys():
            instance_losses["masked_L1"] = self.MaskedL1LossInstances(
                predictions["image"],
                inputs["target"],
                inputs["forward_flow"],
                inputs["backward_flow"],
            )

        if "color_L2" in self.config["losses"].keys():
            instance_losses["color_L2"] = self.MSELossInstances(
                predictions["image"], inputs["target"]
            )

        if "color_gradient" in self.config["losses"].keys():
            instance_losses["color_gradient"] = self.L1LossInstances(
                torch.abs(
                    predictions["image"][..., 1:] - predictions["image"][..., :-1]
                ),
                torch.abs(inputs["target"][..., 1:] - inputs["target"][..., :-1]),
            ) + self.L1LossInstances(
                torch.abs(
                    predictions["image"][..., 1:, :] - predictions["image"][..., :-1, :]
                ),
                torch.abs(inputs["target"][..., 1:, :] - inputs["target"][..., :-1, :]),
            )

        if "flow_smoothness" in self.config["losses"].keys():
            instance_losses["flow_smoothness"] = self.L1LossInstances(
                predictions["flow"][..., 1:], predictions["flow"][..., :-1]
            ) + self.L1LossInstances(
                predictions["flow"][..., 1:, :], predictions["flow"][..., :-1, :]
            )

        if "KL" in self.config["losses"].keys() and "q_means" in predictions:
            instance_losses["KL"] = aggregate_kl_loss(
                predictions["q_means"], predictions["p_means"]
            )

        if "perceptual" in self.config["losses"]:
            instance_losses["perceptual"] = self.PerceptualLossInstances(
                predictions["image"], inputs["target"]
            )

        instance_losses["total"] = sum(
            [
                self.config["losses"][key]["weight"] * instance_losses[key]
                for key in instance_losses.keys()
            ]
        )

        # reduce to batch granularity
        batch_losses = {k: v.mean() for k, v in instance_losses.items()}

        losses = dict(instances=instance_losses, batch=batch_losses)

        return losses

    def prepare_inputs_inplace(self, inputs):
        before = time.time()

        inputs["np"] = {
            "image": inputs["images"][0]["image"],
            "target": inputs["images"][1]["image"],
            "flow": inputs["backward_flow"]
            if self.config["reverse_flow_input"]
            else inputs["forward_flow"],
            "forward_flow": inputs["forward_flow"],
            "backward_flow": inputs["backward_flow"],
        }
        inputs["pt"] = {key: np2pt(inputs["np"][key]) for key in inputs["np"]}

        if retrieve(self.config, "debug_timing", default=False):
            self.logger.info("prepare of data needed {} s".format(time.time() - before))

    def compute_model(self, model, inputs):
        output = model(inputs["pt"])
        predictions = model.warp(output, inputs["pt"])
        return predictions

    def prepare_logs(self, inputs, predictions, losses, model, granularity):
        assert granularity in ["batch", "instances"]
        losses = losses[granularity]

        # prepare simplest benchmark_losses
        # copy input
        predictions_copy_input = {
            "image": inputs["pt"]["image"],
            "flow": inputs["pt"]["flow"],
        }
        losses_copy_input = self.criterion(inputs["pt"], predictions_copy_input)[
            granularity
        ]


        # sample variational part if applicable
        if self.config["model"] == "AnimalPose.models.VUnet":
            output_sample = model(inputs["pt"], mode="sample_appearance")
            predictions_sample = model.warp(output_sample, inputs["pt"])
            losses_sample = self.criterion(inputs["pt"], predictions_sample)[
                granularity
            ]
            sample_images = {
                "images_prediction_sample": pt2np(predictions_sample["image"]),
            }
        else:
            losses_sample = dict()
            sample_images = dict()


        axis = (1, 2, 3) if granularity == "instances" else None


        # mask and masked images
        if "masked_L1" in self.config["losses"]:
            mask, masked_image, masked_target = self.MaskedL1LossInstances.get_masked(
                inputs["pt"]["image"],
                inputs["pt"]["target"],
                inputs["pt"]["forward_flow"],
                inputs["pt"]["backward_flow"],
            )
            mask_images = {
                "mask": pt2np(mask),
                "masked_input": pt2np(masked_image),
                "masked_target": pt2np(masked_target),
            }
        else:
            mask_images = dict()

        # concatenate logs
        logs = {
            "images": {
                "images_input": inputs["np"]["image"],
                "images_target": inputs["np"]["target"],
                "images_prediction": pt2np(predictions["image"]),
                **sample_images,
                **mask_images,
            },
            "scalars": {
                **{"losses/_prediction/" + k: v for k, v in losses.items()},
                **{"losses/copy_input/" + k: v for k, v in losses_copy_input.items()},
                **{"losses/sample/" + k: v for k, v in losses_sample.items()},
            },
        }

        # convert to numpy
        def conditional_convert2np(log_item):
            if isinstance(log_item, torch.Tensor):
                log_item = log_item.detach().cpu().numpy()
            return log_item

        walk(logs, conditional_convert2np, inplace=True)

        return logs

    def step_op(self, model, **inputs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)

        # prepare inputs
        self.prepare_inputs_inplace(inputs)

        # compute model
        predictions = self.compute_model(model, inputs)

        # compute loss
        losses = self.criterion(inputs["pt"], predictions)

        def train_op():
            before = time.time()
            self.optimizer.zero_grad()
            losses["batch"]["total"].backward()
            self.optimizer.step()
            if retrieve(self.config, "debug_timing", default=False):
                self.logger.info("train step needed {} s".format(time.time() - before))

        def log_op():
            with torch.no_grad():
                logs = self.prepare_logs(inputs, predictions, losses, model, "batch")

            # log to tensorboard
            if self.config["integrations"]["tensorboardX"]["active"]:
                if self.get_global_step() == 0 and is_train:
                    # save model
                    self.tensorboardX_writer.add_graph(model, inputs["pt"])
                    self.logger.info("Added model graph to tensorboard")

            return logs

        def eval_op():
            with torch.no_grad():
                logs = self.prepare_logs(
                    inputs, predictions, losses, model, "instances"
                )

            return {
                **logs["images"],
                "labels": {k: v for k, v in logs["scalars"].items()},
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        return {"eval_op": {"cb": eval_callback}}


def eval_callback(root, data_in, data_out, config):
    logger = get_logger("eval_callback")

    prefix = "edeval/target_step_{}/".format(config["target_frame_step"])

    losses = {
        prefix + k.replace("--", "/"): v.mean()
        for k, v in data_out.labels.items()
        if "losses--_prediction" in k
    }

    for k, v in losses.items():
        logger.info("{}: {}".format(k, v))

    if (
        config.get("edeval_update_wandb_summary", True)
        and config["integrations"]["wandb"]["active"]
    ):
        data_out.data.data.root
        import wandb

        api = wandb.Api()

        run_name = data_out.data.data.root.split("/eval/")[0]
        this_run = None
        runs = api.runs("hperrot/flowframegen")
        for run in runs:
            if run.name == run_name:
                this_run = run
        # if this_run is not None:
        #     this_run.summary(losses)

    return losses

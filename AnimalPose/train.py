import torch
import time

import numpy as np
import torch
import torch.optim as optim
from edflow import TemplateIterator, get_logger
from edflow.util import retrieve, walk
import torch.nn.functional

from AnimalPose.hooks.model import RestorePretrainedSDCHook

class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False

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

    def np2pt(self, array):
        tensor = torch.from_numpy(array).float()# torch.tensor(array, dtype=torch.float32)
        if self.cuda:
            tensor = tensor.cuda()
        #tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def pt2np(self, tensor):
        array = tensor.detach().cpu().numpy()
        #array = np.transpose(array, (0, 2, 3, 1))
        return array

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
        #update_loss_weights_inplace(self.config["losses"], self.get_global_step())

        # calculate losses
        instance_losses = {}
        #instance_losses["keypoint_loss"] = MSELoss(inputs, predictions)#batch[f"keypoints{TYPE}"][:, :, :2], coords.double())

        def heatmap_loss(inputs, predictions):
            hm_loss = 0
            for element in range(len(inputs)):
                for idx in range(len(predictions[1])):
                    hm_loss += torch.nn.functional.mse_loss(inputs[element, idx, :, :], predictions[element, idx, :, :])
            return hm_loss
        instance_losses["heatmap_loss"] = heatmap_loss(inputs, predictions)



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

    # def prepare_inputs_inplace(self, inputs):
    #     before = time.time()
    #
    #     inputs["np"] = {
    #         "image": inputs["input"],
    #         "target": inputs["targets"],
    #     }
    #     inputs["pt"] = {key: self.np2pt(inputs["np"][key]) for key in inputs["np"]}
    #
    #     if retrieve(self.config, "debug_timing", default=False):
    #         self.logger.info("Peperation time {} s".format(time.time() - before))

    def compute_model(self, model, inputs):
        output = model(inputs["targets"])
        predictions = output
        return predictions

    # def prepare_logs(self, inputs, predictions, losses, model, granularity):
    #     assert granularity in ["batch", "instances"]
    #     losses = losses[granularity]
    #
    #     # # prepare simplest benchmark_losses
    #     # # copy input
    #     # predictions_copy_input = {
    #     #     "image": inputs["pt"]["image"],
    #     #     "flow": inputs["pt"]["flow"],
    #     # }
    #     # losses_copy_input = self.criterion(inputs["pt"], predictions_copy_input)[
    #     #     granularity
    #     # ]
    #     #
    #     #
    #     # # sample variational part if applicable
    #     # if self.config["model"] == "AnimalPose.models.VUnet":
    #     #     output_sample = model(inputs["pt"], mode="sample_appearance")
    #     #     predictions_sample = model.warp(output_sample, inputs["pt"])
    #     #     losses_sample = self.criterion(inputs["pt"], predictions_sample)[
    #     #         granularity
    #     #     ]
    #     #     sample_images = {
    #     #         "images_prediction_sample": self.pt2np(predictions_sample["image"]),
    #     #     }
    #     # else:
    #     #     losses_sample = dict()
    #     #     sample_images = dict()
    #
    #
    #     #axis = (1, 2, 3) if granularity == "instances" else None
    #
    #
    #     # mask and masked images
    #     # if "masked_L1" in self.config["losses"]:
    #     #     mask, masked_image, masked_target = self.MaskedL1LossInstances.get_masked(
    #     #         inputs["pt"]["image"],
    #     #         inputs["pt"]["target"],
    #     #         inputs["pt"]["forward_flow"],
    #     #         inputs["pt"]["backward_flow"],
    #     #     )
    #     #     mask_images = {
    #     #         "mask": pt2np(mask),
    #     #         "masked_input": pt2np(masked_image),
    #     #         "masked_target": pt2np(masked_target),
    #     #     }
    #     # else:
    #     #     mask_images = dict()
    #
    #     #concatenate logs
    #     logs = {
    #         "images": {
    #             "images_input": inputs["inputs"],
    #             "images_target": inputs["targets"],
    #             "images_prediction": self.pt2np(predictions),
    #             #**sample_images,
    #             #**mask_images,
    #         },
    #         "scalars": {
    #             **{"losses/_prediction/" + k: v for k, v in losses.items()},
    #             #**{"losses/copy_input/" + k: v for k, v in losses_copy_input.items()},
    #             #**{"losses/sample/" + k: v for k, v in losses_sample.items()},
    #         },
    #     }
    #
    #     # convert to numpy
    #     def conditional_convert2np(log_item):
    #         if isinstance(log_item, torch.Tensor):
    #             log_item = log_item.detach().cpu().numpy()
    #         return log_item
    #
    #     walk(logs, conditional_convert2np, inplace=True)
    #
    #     return logs

    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        model.train()

        # prepare inputs
        # self.prepare_inputs_inplace(kwargs)
        # TODO need (batch_size, channel, width, height)
        inp = self.np2pt(kwargs["inp"].reshape(-1, self.config["n_channels"],
                                                      kwargs["inp"].shape[1],
                                                      kwargs["inp"].shape[2]))

        # compute model
        outputs = model(inp)
        # compute loss
        losses = self.criterion(torch.from_numpy(kwargs["targets"]).cpu(), outputs.cpu())

        def train_op():
            before = time.time()
            self.optimizer.zero_grad()
            losses["batch"]["total"].backward()
            self.optimizer.step()
            if retrieve(self.config, "debug_timing", default=False):
                self.logger.info("train step needed {} s".format(time.time() - before))

        def log_op():
            return {
                "images": {"inputs": inp.reshape(-1, 128, 128, 1).cpu().numpy()},
                "scalars": {
                    "heatmap_loss": losses["batch"]["heatmap_loss"],
                    "loss": losses["batch"]["total"],
                },
            }

        def eval_op():
            with torch.no_grad():
                return {
                    "outputs": np.array(outputs.detach().numpy()),
                    "labels": {"loss": np.array(losses["batch"]["total"].detach().numpy())},
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

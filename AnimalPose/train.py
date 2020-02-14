import torch
import time
import pdb
import numpy as np
import torch
import torch.optim as optim
from edflow import TemplateIterator, get_logger
from edflow.util import retrieve, walk
import torch.nn.functional
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy
from AnimalPose.utils.loss_utils import heatmap_loss, keypoint_loss
from AnimalPose.data.util import heatmap_to_image
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
        # calculate losses
        instance_losses = {}
        if self.config["losses"]["L2"]:
            instance_losses["heatmap_loss"] = heatmap_loss(targets, predictions)
        if self.config["losses"]["L2_kpt"]:
            instance_losses["keypoint_loss"] = keypoint_loss(predictions, gt_coords)
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

        # prepare inputs
        # self.prepare_inputs_inplace(kwargs)

        # TODO need (batch_size, channel, width, height)
        # kwargs["inp"]
        # (batch_size, width, height, channel)
        inputs = numpy2torch(kwargs["inp"].transpose(0,3,1,2)).to("cuda")
        # inputs now
        # (batch_size, channel, width, height)

        # compute model
        predictions = model(inputs)
        # compute loss
        # Target heatmaps, predicted heatmaps, gt_coords
        losses = self.criterion(kwargs["targets"], predictions.cpu(), kwargs["kps"])

        def train_op():
            before = time.time()
            self.optimizer.zero_grad()
            losses["batch"]["total"].backward()
            self.optimizer.step()
            if retrieve(self.config, "debug_timing", default=False):
                self.logger.info("train step needed {} s".format(time.time() - before))

        def log_op():
            from AnimalPose.utils.log_utils import plot_input_target_keypoints
            logs = {
                "images": {
                    "image_input": torch2numpy(inputs).transpose(0,2,3,1),
                    "outputs": heatmap_to_image(torch2numpy(predictions)).transpose(0,2,3,1),
                    "targets": heatmap_to_image(kwargs["targets"]).transpose(0,2,3,1),

                },
                "scalars": {
                    "loss": losses["batch"]["total"],
                },
                "figures": {
                    "Keypoint Mapping": plot_input_target_keypoints(torch2numpy(inputs).transpose(0,2,3,1), # get BHWC
                                                        torch2numpy(predictions),# stay BCHW
                                                        kwargs["kps"]),
                }
            }
            if self.config["losses"]["L2"]:
                logs["scalars"]["heatmap_loss"] = losses["batch"]["heatmap_loss"]
            if self.config["losses"]["L2_kpt"]:
                logs["scalars"]["keypoint_loss"]: losses["batch"]["keypoint_loss"]

            # log to tensorboard
            #if self.config["integrations"]["tensorboard"]["active"]:
            #    # save model
            #    self.tensorboard_writer.add_graph(model)
            #    self.logger.info("Added model graph to tensorboard")


            return logs

        def eval_op():
            #return {
                #"outputs": np.array(predictions.cpu().detach().numpy()),
                #TODO in which shape is the outputs necessary for evaluation?
                #"labels": {k: [v.cpu().detach().numpy()] for k, v in losses["batch"].items()},
            #}
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

#     @property
#     def callbacks(self):
#         return {"eval_op": {"cb": eval_callback}}
#
#
# def eval_callback(root, data_in, data_out, config):
#     logger = get_logger("eval_callback")
#
#     prefix = "edeval/target_step_{}/".format(config["target_frame_step"])
#
#     losses = {
#         prefix + k.replace("--", "/"): v.mean()
#         for k, v in data_out.labels.items()
#         if "losses--_prediction" in k
#     }
#
#     for k, v in losses.items():
#         logger.info("{}: {}".format(k, v))
#
#     if (
#         config.get("edeval_update_wandb_summary", True)
#         and config["integrations"]["wandb"]["active"]
#     ):
#         data_out.data.data.root
#         import wandb
#
#         api = wandb.Api()
#
#         run_name = data_out.data.data.root.split("/eval/")[0]
#         this_run = None
#         runs = api.runs("hperrot/flowframegen")
#         for run in runs:
#             if run.name == run_name:
#                 this_run = run
#         # if this_run is not None:
#         #     this_run.summary(losses)
#
#     return losses

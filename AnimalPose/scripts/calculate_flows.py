import os
import skimage.transform
import torch
from edflow import TemplateIterator, get_logger
import flowiz as fz
from AnimalPose.train import np2pt, pt2np
from AnimalPose.models.flownet2_pytorch.utils.flow_utils import writeFlow


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self)

        # restore checkpoint
        checkpoint_path = self.config.get(
            "checkpoint_path", "checkpoints/FlowNet2_checkpoint.pth.tar"
        )
        self.restore(checkpoint_path)

        # set to eval
        self.model.eval()

        # load to gpu
        if torch.cuda.is_available():
            self.model.cuda()

    def save(self, checkpoint_path):
        pass

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["state_dict"])
        self.logger.info("restored checkpoint from {}".format(checkpoint_path))

    def prepare_inputs_inplace(self, inputs):

        if not self.config["reverse_flow_input"]:
            inputs["np"] = {
                "image": inputs["images"][0]["image"],
                "target": inputs["images"][1]["image"],
            }
        else:
            inputs["np"] = {
                "image": inputs["images"][1]["image"],
                "target": inputs["images"][0]["image"],
            }
        inputs["pt"] = {key: np2pt(inputs["np"][key]) for key in inputs["np"]}

    def step_op(self, model, **inputs):

        # prepare inputs
        self.prepare_inputs_inplace(inputs)

        # compute outputs
        with torch.no_grad():
            outputs = self.model(inputs["pt"])
            outputs_np = pt2np(outputs)

        # write outputs to disk
        for i in range(len(outputs_np)):
            flow = outputs_np[i]
            flow = skimage.transform.resize(flow, [self.config["load_size"]] * 2)
            if not self.config["reverse_flow_input"]:
                file_path = str(inputs["labels_"]["forward_flow_"][i])
            else:
                file_path = str(inputs["labels_"]["backward_flow_"][i])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            writeFlow(file_path, flow)

        def train_op():
            return dict()

        def log_op():
            return dict()

        def eval_op():
            return dict()

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

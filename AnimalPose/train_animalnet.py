import time

import torch
import torch.nn.functional
import torch.optim as optim
from edflow import TemplateIterator
from edflow.util import retrieve

from AnimalPose.utils.log_utils import hog_similarity, hist_similarity, generate_samples
from AnimalPose.utils.tensor_utils import numpy2torch, torch2numpy
from AnimalPose.utils.perceptual_loss.models import PerceptualLoss
from AnimalPose.utils.LossConstrained import LossConstrained
from AnimalPose.models.resnet import ResnetTorchVisionClass
import numpy as np
from edflow.data.util import adjust_support
from edflow.util import retrieve


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
        if retrieve(self.config, "classifier/active", default=False):
            self.classifier = ResnetTorchVisionClass(self.config).to(self.device)
            # Load pretrained classifier into memory
            self.logger.info(f"Loading weights for classifier from {self.config['classifier']['weights']}")
            state_dict = torch.load(self.config["classifier"]["weights"], map_location=self.device)["model"]
            self.classifier.load_state_dict(state_dict)

        try:
            self.start_step, self.stop_step, self.start_weight, self.stop_weight = self.config["variational"][
                                                                                       "start_step"], \
                                                                                   self.config[
                                                                                       "num_steps"], self.kl_weight, \
                                                                                   self.config["variational"][
                                                                                       "stop_weight"]
        except:
            print("Some infos not found")
        self.loss_constrained = LossConstrained(self.config)
        self.sigmoid = torch.nn.Sigmoid()
        # # initalize perceptual loss if possible
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=False).to(
                self.device)
        if self.cuda:
            self.model.cuda()

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
            else:
                predictions = model(inputs0, inputs1)

            with torch.no_grad():
                inputs1_flipped = torch.flip(inputs1, [0])  # flip the tensor in zero dimension
                if self.variational:
                    kl_test_preds, _, _ = model(inputs0, inputs1_flipped)
                else:
                    kl_test_preds = model(inputs0, inputs1_flipped)
                kl_test_preds = self.sigmoid(kl_test_preds)
                if retrieve(self.config, "classifier/active", default=False):
                    class_prediction = self.classifier(kl_test_preds)
                    _, preds = torch.max(class_prediction, 1)
                    # as inputs1 is flipped, the labels need also to be flipped
                    labels = torch.flip(torch.from_numpy(kwargs["global_video_class1"]).to("cuda"), [0])
                    # Compute accuracy for batch
                    corrects = torch.sum(preds == labels.data)
                    accuracy = corrects.double() / self.config["batch_size"]
        else:
            if self.variational:
                predictions, mu, logvar = model(inputs0)
            else:
                predictions = model(inputs0)
        predictions = self.sigmoid(predictions)  # in order to make the output in range between 0 and 1
        # compute loss
        # Target heatmaps, predicted heatmaps, gt_coords
        if self.get_global_step() >= self.config["LossConstrained"]["no_kl_for"] and \
                self.config["LossConstrained"]["active"]:
            loss, log, loss_train_op = self.loss_constrained(kwargs["inp0"].transpose(0, 3, 1, 2), predictions, mu,
                                                             logvar,
                                                             self.get_global_step())
            if is_train:
                loss_train_op()
        else:
            loss = torch.mean(self.perceptual_loss(inputs0, predictions, True))

        hist_values = []
        hog_values = []
        for idx in range(predictions.size(0)):
            hist_values.append(hist_similarity(kl_test_preds[idx].cpu().numpy().transpose(1, 2, 0),
                                               inputs1_flipped[idx].cpu().numpy().transpose(1, 2, 0)))
            hog_values.append(hog_similarity(inputs0[idx].cpu().numpy().transpose(1, 2, 0),
                                             kl_test_preds[0].cpu().numpy().transpose(1, 2, 0))[0])

        def train_op():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():
            logs = {
                "images": {
                    "image_input_0": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "disentanglement": adjust_support(np.expand_dims(generate_samples(inputs0, model),0), "-1->1", "0->1"),
                    "outputs": adjust_support(torch2numpy(predictions).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                },
                "scalars": {
                    "loss": loss,
                },
            }
            logs["images"]["image_input_1_flipped"] = adjust_support(
                torch2numpy(inputs1_flipped).transpose(0, 2, 3, 1), "-1->1", "0->1")
            if self.get_global_step() >= self.config["LossConstrained"]["no_kl_for"] and \
                    self.config["LossConstrained"]["active"]:
                logs["scalars"]["lambda_"] = log["scalars"]["lambda_"]
                logs["scalars"]["gain"] = log["scalars"]["gain"]
                logs["scalars"]["active"] = log["scalars"]["active"]
                logs["scalars"]["kl_loss"] = log["scalars"]["kl_loss"]
                logs["scalars"]["nll_loss"] = log["scalars"]["nll_loss"]
                logs["scalars"]["rec_loss"] = log["scalars"]["rec_loss"]
                logs["scalars"]["mu"] = log["scalars"]["mu"]
                logs["scalars"]["eps"] = log["scalars"]["eps"]

            if self.encoder_2:
                logs["images"]["image_input_1"] = adjust_support(torch2numpy(inputs1).transpose(0, 2, 3, 1), "-1->1",
                                                                 "0->1")
                logs["images"]["image_input_1_flipped"] = adjust_support(
                    torch2numpy(inputs1_flipped).transpose(0, 2, 3, 1), "-1->1", "0->1")
                logs["images"]["pose_reconstruction"] = adjust_support(
                    torch2numpy(kl_test_preds).transpose(0, 2, 3, 1), "-1->1", "0->1")
                if retrieve(self.config, "classifier/active", default=False):
                    logs["scalars"]["accuracy"] = accuracy
            if self.encoder_2:
                logs["scalars"]["hog"] = np.array(hog_values).mean()
                logs["scalars"]["hog_std"] = np.array(hog_values).std()
                logs["scalars"]["hist"] = np.array(hist_values).mean()
                logs["scalars"]["hist_std"] = np.array(hist_values).std()

            return logs

        def eval_op():
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


class TripletIterator(Iterator):
    def step_op(self, model, **kwargs):
        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)
        # inp0 -> p0a0 (a)
        # inp1 -> p0a1 (b)
        # inp2 -> p1a0 (c)
        # inp3 -> p1a1 (d)
        # (batch_size, width, height, channel)

        inputs0 = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2)).to("cuda")
        if self.encoder_2:
            inputs2 = numpy2torch(kwargs["inp2"].transpose(0, 3, 1, 2)).to("cuda")
        # (batch_size, channel, width, height)
        # compute model
        if self.encoder_2:
            if self.variational:
                predictions, mu, logvar = model(inputs0, inputs2)
            else:
                predictions = model(inputs0, inputs2)
        else:
            if self.variational:
                predictions, mu, logvar = model(inputs0)
            else:
                predictions = model(inputs0)
        # in order to make the output in range between 0 and 1
        predictions = self.sigmoid(predictions)
        # compute loss
        # Target heatmaps, predicted heatmaps, gt_coords
        loss, log, loss_train_op = self.loss_constrained(kwargs["inp0"].transpose(0, 3, 1, 2),
                                                         predictions, mu, logvar,
                                                         self.get_global_step())
        if is_train:
            loss_train_op()
        # compute perceptual loss reconstruction
        # loss(predictions, inp1) where inp2 is p1a1
        # Testing:
        # inp0 -> p0a0 (a)
        # inp1 -> p0a1 (b)
        # inp2 -> p1a0 (c)
        # inp3 -> p1a1 (d)
        with torch.no_grad():
            inputs3 = numpy2torch(kwargs["inp3"].transpose(0, 3, 1, 2)).to("cuda")  # p1a1
            inputs1 = numpy2torch(kwargs["inp1"].transpose(0, 3, 1, 2)).to("cuda")  # p0a1
            testing, _, _ = model(inputs0, inputs3)
            testing = self.sigmoid(testing)
            loss_dis = torch.mean(self.perceptual_loss(testing, inputs1))

        def train_op():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():

            logs = {
                "images": {
                    "p0a0": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "p0a1": adjust_support(torch2numpy(inputs1).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "p1a0": adjust_support(torch2numpy(inputs2).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "p1a1": adjust_support(torch2numpy(inputs3).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "disentanglement": adjust_support(torch2numpy(predictions).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "testing": adjust_support(torch2numpy(testing).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                },
                "scalars": {
                    "loss": loss,
                    "loss_dis": loss_dis,
                },
            }
            logs["scalars"]["lambda_"] = log["scalars"]["lambda_"]
            logs["scalars"]["gain"] = log["scalars"]["gain"]
            logs["scalars"]["active"] = log["scalars"]["active"]
            logs["scalars"]["kl_loss"] = log["scalars"]["kl_loss"]
            logs["scalars"]["nll_loss"] = log["scalars"]["nll_loss"]
            logs["scalars"]["rec_loss"] = log["scalars"]["rec_loss"]
            logs["scalars"]["mu"] = log["scalars"]["mu"]
            logs["scalars"]["eps"] = log["scalars"]["eps"]

            return logs

        def eval_op():
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


class ReconstructionIterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.cuda = True if self.config["cuda"] and torch.cuda.is_available() else False
        self.device = "cuda" if self.cuda else "cpu"
        self.variational = False
        self.encoder_2 = False
        self.sigmoid = torch.nn.Sigmoid()

        # # initalize perceptual loss if possible
        if self.config["losses"]["perceptual"]:
            net = self.config["losses"]["perceptual_network"]
            assert net in ["alex", "squeeze",
                           "vgg"], f"Perceptual network needs to be 'alex', 'squeeze' or 'vgg', got {net}"
            self.perceptual_loss = PerceptualLoss(model='net-lin', net=net, use_gpu=self.cuda, spatial=False).to(
                self.device)
        if self.cuda:
            self.model.cuda()

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
        batch_losses = {}
        if self.config["losses"]["L2"]:
            batch_losses["L2_loss"] = crit(torch.from_numpy(targets), predictions.cpu()).to(self.device)

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

        # (batch_size, width, height, channel)
        inputs0 = numpy2torch(kwargs["inp0"].transpose(0, 3, 1, 2)).to("cuda")
        # compute model

        predictions = model(inputs0)
        predictions = self.sigmoid(predictions)  # in order to make the output in range between 0 and 1
        # compute loss

        loss = torch.mean(self.perceptual_loss(inputs0, predictions))

        def train_op():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():
            is_train = self.get_split() == "train"
            logs = {
                "images": {
                    "input": adjust_support(torch2numpy(inputs0).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                    "outputs": adjust_support(torch2numpy(predictions).transpose(0, 2, 3, 1), "-1->1", "0->1"),
                },
                "scalars": {
                    "loss": loss,
                },
            }
            return logs

        def eval_op():
            return

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

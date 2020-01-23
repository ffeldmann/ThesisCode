import torch

from edflow import get_logger
from edflow.hooks.hook import Hook


class RestorePretrainedSDCHook(Hook):
    """Restores pretrained SDC Network for all layers whose shape fits"""

    def __init__(self, pretrained_checkpoint, model):
        self.model = model
        self.pretrained_checkpoint = pretrained_checkpoint
        self.logger = get_logger(self)

    def before_epoch(self, epoch):
        if epoch == 0:
            checkpoint_dict = torch.load(self.pretrained_checkpoint)
            if "state_dict" in checkpoint_dict:  # original SDC checkpoints
                state_dict_checkpoint = checkpoint_dict["state_dict"]
            elif "model" in checkpoint_dict:  # own trained SDC checkpoints
                state_dict_checkpoint = checkpoint_dict["model"]
            else:
                assert False
            state_dict_model = self.model.state_dict()
            incompatible_keys = []
            for key in state_dict_checkpoint.keys():
                if not (
                    (key in state_dict_model.keys())
                    and (
                        state_dict_checkpoint[key].size()
                        == state_dict_model[key].size()
                    )
                ):
                    incompatible_keys.append(key)
            for key in incompatible_keys:
                del state_dict_checkpoint[key]
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict_checkpoint, strict=False
            )
            self.logger.info(
                "Restored pretrained SDC-Model from checkpoint {}".format(
                    self.pretrained_checkpoint
                )
            )
            self.logger.info(
                "Removed keys from checkpoint {}".format(incompatible_keys)
            )
            self.logger.info("Missing keys {}".format(missing_keys))
            self.logger.info("Unexpected keys {}".format(unexpected_keys))


class TrainHeadTailFirstNHook(Hook):
    """
    Trains only head and tail of network for first n iterations.
    """

    def __init__(self, model, n, layers_to_train=None):
        self.model = model
        self.n = n
        if layers_to_train is None:
            self.layers_to_train = ["conv1", "final_flow"]
        else:
            self.layers_to_train = layers_to_train
        self._locked = None
        self.logger = get_logger(self)
        self.pre_locked_state = {}

    def before_step(self, step, fetches, feeds, batch):
        def log_trainable_state():
            trainable_dict = {
                key: param.requires_grad for key, param in self.model.named_parameters()
            }
            locked = [key for key in trainable_dict if trainable_dict[key] is False]
            trainable = [key for key in trainable_dict if trainable_dict[key] is True]
            self.logger.info("Trainable parameters {}".format(trainable))
            self.logger.info("Not trainable parameters {}".format(locked))

        # It is enough to set requires_grad to fallse for the parts of the network that should not be trained
        # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/22
        if (step < self.n) and (self._locked is not True):
            self._locked = True
            # lock training for all except head and tail
            for name, param in self.model.named_parameters():
                self.pre_locked_state.update({name: param.requires_grad})
                to_train = param.requires_grad and any(
                    [
                        name.startswith(layer_to_train)
                        for layer_to_train in self.layers_to_train
                    ]
                )
                param.requires_grad = to_train
            self.logger.info("Locked bulk of network")
            log_trainable_state()

        if (step >= self.n) and (self._locked is not False):
            self._locked = False
            # resume training for all trainable parameters of model
            for name, param in self.model.named_parameters():
                param.requires_grad = self.pre_locked_state[name]
            self.logger.info("Unlocked bulk of network")
            log_trainable_state()

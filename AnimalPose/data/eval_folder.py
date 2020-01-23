import numpy as np
from edflow.eval.pipeline import EvalDataFolder
from edflow.data.util import adjust_support
from edflow.util import walk


class EvalFolderEdexplore(EvalDataFolder):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config

    def get_example(self, idx):
        ex = super().get_example(idx)

        def conditional_adjust(value):
            if (
                isinstance(value, np.ndarray)
                and len(value.shape) == 3
                and value.shape[2] == 3
            ):
                value = adjust_support(value, self.config["support"])
            return value

        walk(ex, conditional_adjust, inplace=True)
        return ex


if __name__ == "__main__":
    config = {
        "dataroot": "logs/2020-01-16T18-51-08_meta/eval/2020-01-16T18-51-08_meta/11",
        "support": "-1->1",
    }
    ds = EvalFolderEdexplore(config)

    ex = ds[0]

    pass

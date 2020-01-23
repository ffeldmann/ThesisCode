import ffmpeg
import glob
import os
import sys
import flowiz as fz
import tqdm
import numpy as np
import argparse
import matplotlib

sys.path.append(os.path.dirname("."))
from AnimalPose.data.util import get_flow_folder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "flow_calc_mode", action="store", help="Flow calc mode: flownet2, pwc-net, open_cv"
)
parser.add_argument(
    "target_frame_step", action="store", type=int, help="Flow step: 1, 5, 10, 25"
)
parser.add_argument(
    "--base_frame_step",
    action="store",
    type=int,
    default=10,
    help="Base frames step, default 10",
)
parser.add_argument(
    "--only_examples",
    action="store",
    type=bool,
    default=True,
    help="only visualize examples, default True",
)


args = parser.parse_args()


class Options:
    def __init__(self, **entries):
        self.__dict__.update(entries)


example_person = "S1"
examples = [
    ("Eating", "1", "55011271"),
    ("Phoning", "2", "58860488"),
    ("WalkingDog", "2", "54138969"),
    ("Greeting", "2", "58860488"),
    ("Walking", "1", "54138969"),
    ("WalkingTogether", "1", "60457274"),
    ("Directions", "1", "54138969"),
]


if __name__ == "__main__":

    opt = Options(
        flow_calc_mode=args.flow_calc_mode,  # [pwc_net, flownet2, open_cv]
        dataroot="/export/scratch/hperrot/Datasets/human3.6M",
        flow_keep_intermediate=False,
        reverse_flow_input=True,
        target_frame_step=args.target_frame_step,
        base_frame_step=args.base_frame_step,
        load_size=256,
        calc_size=1024,
        max_dataset_size=sys.maxsize,
    )
    original_video_framerate = 50

    flow_root = os.path.join(
        opt.dataroot, "flow", get_flow_folder(opt.__dict__, opt.reverse_flow_input)
    )
    print("flow_root:", flow_root)

    flow_files = glob.glob(flow_root + "/**/*.flo", recursive=True)
    flow_folders = sorted(set([os.path.dirname(file) for file in flow_files]))

    for flow_folder in tqdm.tqdm(flow_folders, total=len(flow_folders)):
        # figure out names of files and folders
        person = flow_folder.split("/")[-3]
        action = flow_folder.split("/")[-2]
        camera = flow_folder.split("/")[-1]

        if args.only_examples:
            if person != example_person:
                continue
            action_only, subaction = action.split("-")
            if (action_only, subaction, camera) not in examples:
                continue
            print("example to visualize: ", flow_folder)

        flow_video_folder = os.path.join(
            flow_root.replace("/flow/", "/visualized/flow_"), person
        )
        flow_png_folder = os.path.join(flow_video_folder, action + "_" + camera)
        flow_files = [
            filename
            for filename in glob.glob(flow_folder + "/*.flo")
            if (
                int(os.path.basename(filename).replace("flow_", "").replace(".flo", ""))
                - 1
            )
            % opt.base_frame_step
            == 0
        ]

        # create folder
        os.makedirs(flow_png_folder, exist_ok=True)

        # get max flow value per video and flow statistics
        mins = []
        maxs = []
        means = []
        max_rad = 0.0
        for flow_file in tqdm.tqdm(
            flow_files, total=len(flow_files), desc="computing max flow"
        ):
            flow = fz.read_flow(flow_file)
            u = flow[:, :, 0]
            v = flow[:, :, 1]
            rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
            mins.append(np.min(rad))
            maxs.append(np.max(rad))
            means.append(np.mean(rad))
        max_rad = max(maxs)

        # plot min, max, mean over video
        dt = float(args.base_frame_step) / float(original_video_framerate)
        eps = 1e-7
        t = np.arange(0, len(mins) * dt - eps, dt)
        plt.figure()
        plt.plot(t, mins, label="min")
        plt.plot(t, maxs, label="max")
        plt.plot(t, means, label="mean")
        plt.xlabel("time [s]")
        plt.ylabel("flow radius")
        plt.legend()
        plt.savefig(flow_png_folder + ".svg")
        # convert to png
        fz.convert_files(flow_files, outdir=flow_png_folder, min_rad=max_rad)

        # convert to mp4
        (
            ffmpeg.input(
                flow_png_folder + "/*.png", pattern_type="glob", framerate=1 / dt
            )
            .output(flow_png_folder + ".mp4")
            .overwrite_output()
            .run()
        )

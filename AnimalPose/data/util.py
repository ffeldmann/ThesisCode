import os
import numpy as np


def get_flow_folder(config, reverse_flow_input):
    return os.path.join(
        config["flow_calc_mode"],
        ("r_" if reverse_flow_input else "")
        + str(config["target_frame_step"])
        + "_"
        + str(config["load_size"])
        + "_"
        + str(config["calc_size"]),
    )


def num_frames(labels):
    one_key = next(iter(labels.keys()))
    return len(labels[one_key])


def get_frames_split(labels, split_type, mode):
    def get_split_indices(mode):
        assert split_type in ["person", "action", "none"]

        if split_type == "person":
            labels_to_split = labels["subject"]
        else:  # split_type == 'action'
            labels_to_split = labels["action"]

        items_unique = sorted(set(labels_to_split))
        num_train = int(len(items_unique) * 0.8)  # reserve at least 20% for evaluation

        if mode == "train":
            split_items = items_unique[:num_train]
        else:  # mode == 'validation'
            split_items = items_unique[num_train:]

        # keep only items of train or validation dataset
        split_indices = np.where(np.isin(labels_to_split, split_items))[0]

        return split_indices

    assert mode in ["train", "validation", "all"]
    if mode == "all":
        return np.arange(num_frames(labels))
    elif mode == "train":
        return get_split_indices("train")
    elif mode == "validation":
        return (get_split_indices("validation"),)


def split_sequences(split_indices, sequences):

    split_sequence_ids = np.where(np.isin(sequences[:, 0], split_indices))
    split_sequences = sequences[split_sequence_ids]

    return split_sequences


def get_sequences(labels, base_frame_step, target_frame_step):
    def same_video(index1, index2):
        path1 = labels["frame_path"][index1]
        path2 = labels["frame_path"][index2]
        return os.path.dirname(path1) == os.path.dirname(path2)

    fids = labels["fid"]
    new_video_indices = np.where(np.append(np.inf, fids[:-1]) > fids)[0]

    base_indices = np.array([], dtype=np.int64)
    for i in range(len(new_video_indices)):
        if i < len(new_video_indices) - 1:
            stop = new_video_indices[i + 1]
        else:
            stop = len(fids)
        new_base_indices = np.arange(
            start=new_video_indices[i], stop=stop, step=base_frame_step
        )
        base_indices = np.append(base_indices, new_base_indices)

    # base_indices = np.arange(start=0, stop=num_frames(), step=base_frame_step)
    target_indices = base_indices + target_frame_step

    # elliminate too big indices
    good_indices = np.where(target_indices < num_frames(labels))[0]
    base_indices = base_indices[good_indices]
    target_indices = target_indices[good_indices]

    # elliminate not same video
    good_indices = np.where(
        [same_video(*indices) for indices in zip(base_indices, target_indices)]
    )[0]
    base_indices = base_indices[good_indices]
    target_indices = target_indices[good_indices]

    sequences = np.stack([base_indices, target_indices], axis=1)

    return sequences

"""Dataset discovery and dataloader construction."""

from .common import *


class SimpleBaseDataset:
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def make_infrared_dataset_list(root):
    if not os.path.exists(root):
        print(f"Dataset root {root} does not exist!")
        return []
    files = sorted(os.listdir(root))
    rgb_files = {f.split('_RGB')[0]: f for f in files if '_RGB.jpg' in f}
    ir_files = {f.split('_PreviewData')[0]: f for f in files if '_PreviewData.jpeg' in f}

    data = []
    keys = sorted(list(set(rgb_files.keys()) & set(ir_files.keys())))
    print(f"Found {len(keys)} paired images in {root}")
    for k in keys:
        data.append({
            "infrared_path": ir_files[k],
            "vision_path": rgb_files[k],
            "description_vision_only": "",
            "description_instruct": ""
        })
    return data


def build_infrared_loader(data_dir, batch_size, num_workers, input_size, split="train", args=None):
    if split == "train":
        json_path = args.train_file

        implicit_path = os.path.join(data_dir, "train_set.json")
        if not os.path.exists(implicit_path):
            implicit_path = os.path.join(data_dir, "annotations_train.json")
    else:
        json_path = args.val_file

        implicit_path = os.path.join(data_dir, "test_set.json")
        if not os.path.exists(implicit_path):
            implicit_path = os.path.join(data_dir, "annotations_validation.json")

    final_json_path = None
    if json_path and os.path.exists(json_path):
        final_json_path = json_path
    elif os.path.exists(implicit_path):
        final_json_path = implicit_path

    if final_json_path:
        print(f"Loading {split} dataset from {final_json_path}")
        with open(final_json_path, "r") as f:
            data_list = json.load(f)
    else:
        print(f"Warning: JSON path not found (provided: '{json_path}', implicit: '{implicit_path}'), falling back to directory scan.")
        data_list = make_infrared_dataset_list(data_dir)
        random.seed(42)
        random.shuffle(data_list)
        split_idx = int(0.9 * len(data_list))
        if split == "train":
            data_list = data_list[:split_idx]
        else:
            data_list = data_list[split_idx:]

    if not data_list:
        raise FileNotFoundError(
            f"No {split} samples found under {data_dir}; check the dataset "
            "directory and its split manifests."
        )

    base_ds = SimpleBaseDataset(data_list)
    ds = InfraredDataset(base_ds, dataset_root=data_dir, condition_size=input_size, target_size=input_size, resize_resolution=input_size)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=(split=="train"), pin_memory=True, drop_last=(split=="train"))

"""Evaluation metrics for generated infrared images."""

import argparse
import json
import os
import shutil

import lpips
import torch
import torchvision
from cleanfid import fid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm


class InfraredDataset(Dataset):
    def __init__(self, data_root, json_path, generated_path, resolution=(512, 384)):
        self.data_root = data_root
        self.generated_path = generated_path
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.resolution = resolution
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.resolution),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        real_path = os.path.join(self.data_root, item["infrared_path"])
        gen_name = os.path.basename(item["infrared_path"])
        gen_path = os.path.join(self.generated_path, gen_name)

        real_img = Image.open(real_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        real_tensor, gen_tensor = self.transform(real_img), self.transform(gen_img)
        return real_tensor, gen_tensor, item


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_lpips_ssim_psnr(
    data_root,
    json_path,
    generated_path,
    output_file,
    batch_size=1,
    resolution=(512, 384),
):
    dataset = InfraredDataset(data_root, json_path, generated_path, resolution)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    ssim_metric = SSIM(data_range=1.0).to(device)
    total_lpips, total_ssim, total_psnr = 0.0, 0.0, 0.0
    total_samples = len(dataset)

    with open(output_file, "w") as f:
        f.write("|Filename|PSNR|SSIM|LPIPS|\n")
        f.write("|-----------|-------------|--------------|--------------|\n")

    with torch.no_grad():
        for _, (real_imgs, gen_imgs, items) in enumerate(
            tqdm(dataloader, desc="计算指标")
        ):
            real_imgs = real_imgs.to(device).float()
            gen_imgs = gen_imgs.to(device).float()
            lpips_score = lpips_model.forward(real_imgs * 2 - 1, gen_imgs * 2 - 1)
            ssim_score = ssim_metric(gen_imgs, real_imgs)
            psnr_score = psnr(real_imgs, gen_imgs)

            total_lpips += lpips_score.sum().item()
            total_ssim += ssim_score.sum().item()
            total_psnr += (
                psnr_score.item()
                if real_imgs.size(0) == 1
                else psnr_score.sum().item()
            )
            assert batch_size == 1
            filename = os.path.basename(items["infrared_path"][0])
            single_lpips = lpips_score[0].item()
            single_ssim = ssim_score.item()
            single_psnr = psnr_score.item()
            with open(output_file, "a") as f:
                f.write(
                    f"|{filename}|{single_psnr:.4f}|{single_ssim:.4f}|"
                    f"{single_lpips:.4f}|\n"
                )

    avg_lpips = total_lpips / total_samples
    avg_ssim = total_ssim / total_samples
    avg_psnr = total_psnr / total_samples
    with open(output_file, "a") as f:
        f.write(f"|avg|{avg_psnr:.4f}|{avg_ssim:.4f}|{avg_lpips:.4f}|\n")
    print(f"平均指标: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
    return avg_psnr, avg_ssim, avg_lpips


def calculate_fid(real_path, generated_path, output_file):
    print("计算 FID 指标...")
    fid_clean_v3 = fid.compute_fid(
        generated_path,
        real_path,
        mode="clean",
        model_name="inception_v3",
        num_workers=4,
        batch_size=1,
    )
    with open(output_file, "a") as f:
        f.write("\n|FID_clean_v3|\n")
        f.write("|-----------|\n")
        f.write(f"|{fid_clean_v3:.4f}|\n")
    print(f"FID_clean_v3={fid_clean_v3:.4f}")
    return fid_clean_v3


def create_temp_test_dir(json_path: str, data_root: str, temp_test_dir: str) -> None:
    """Copy test infrared images listed in a JSON manifest to a temporary folder."""
    print(f"读取标注文件: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    test_samples = annotations
    if not test_samples:
        raise ValueError("未从JSON中找到test_set样本！请检查JSON的split字段或筛选逻辑")
    print(f"找到 {len(test_samples)} 个测试集样本")

    os.makedirs(temp_test_dir, exist_ok=True)
    for old_file in os.listdir(temp_test_dir):
        old_file_path = os.path.join(temp_test_dir, old_file)
        if os.path.isfile(old_file_path):
            os.remove(old_file_path)

    print(f"复制测试集图像到临时文件夹: {temp_test_dir}")
    missing_count = 0
    for idx, sample in enumerate(test_samples):
        img_rel_path = sample.get("infrared_path")
        if not img_rel_path:
            print(f"警告：样本 {idx} 缺少红外图像路径，跳过")
            missing_count += 1
            continue

        img_abs_path = os.path.join(data_root, img_rel_path)
        if not os.path.exists(img_abs_path):
            print(f"警告：图像文件不存在 {img_abs_path}，跳过")
            missing_count += 1
            continue

        img_filename = os.path.basename(img_abs_path)
        dest_path = os.path.join(temp_test_dir, img_filename)
        shutil.copy2(img_abs_path, dest_path)

    if missing_count > 0:
        print(f"总计 {missing_count} 个样本的图像缺失或路径错误")
    print(f"临时测试集文件夹准备完成，包含 {len(os.listdir(temp_test_dir))} 张图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated infrared images.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--generated_path", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--temp_test_dir", default="./temp_test_dir")
    parser.add_argument("--resolution", nargs=2, type=int, default=(512, 640))
    args = parser.parse_args()

    output_file = args.output_file or os.path.join(
        args.generated_path, "evaluation_results.md"
    )
    try:
        calculate_lpips_ssim_psnr(
            args.data_root,
            args.json_path,
            args.generated_path,
            output_file,
            resolution=tuple(args.resolution),
        )
        create_temp_test_dir(args.json_path, args.data_root, args.temp_test_dir)
        calculate_fid(args.temp_test_dir, args.generated_path, output_file)
    finally:
        if os.path.exists(args.temp_test_dir):
            print(f"Cleaning temporary directory: {args.temp_test_dir}")
            shutil.rmtree(args.temp_test_dir)
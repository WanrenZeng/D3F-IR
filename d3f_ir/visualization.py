"""Training and feature visualization helpers."""

from .common import *


def dino_v2_visualize(
    patch_tokens: torch.Tensor,
    img_size: Union[int, Tuple[int, int]] = 512,
    patch_size: int = 16,
) -> np.ndarray:
    if patch_tokens.dim() == 3:
        tokens = patch_tokens[0]
    else:
        tokens = patch_tokens

    if isinstance(img_size, int):
        h_grid = w_grid = img_size // patch_size
        target_size = (img_size, img_size)
    else:
        h_grid = img_size[0] // patch_size
        w_grid = img_size[1] // patch_size
        target_size = (img_size[1], img_size[0])


    num_tokens = tokens.shape[0]


    if num_tokens != h_grid * w_grid:


        if num_tokens == 1280:
             h_grid, w_grid = 32, 40
        else:
             side = int(math.sqrt(num_tokens))
             if side * side == num_tokens:
                 h_grid = side
                 w_grid = side


    if tokens.shape[0] > 100 and tokens.shape[0] != h_grid * w_grid:

         tokens = tokens[- (h_grid * w_grid) :]

    flat = tokens.detach().cpu()
    if flat.shape[0] != h_grid * w_grid:

         print(f"Viz shape mismatch: {flat.shape[0]} vs {h_grid}x{w_grid}")
         return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    features_flat = flat.reshape(-1, flat.shape[-1]).numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)
    pca_min = pca_features.min(0)
    pca_range = pca_features.max(0) - pca_min
    pca_rgb = (pca_features - pca_min) / (pca_range + 1e-6)
    pca_rgb = pca_rgb.reshape(h_grid, w_grid, 3)
    to_image = transforms.ToPILImage()


    upscaled_pil = transforms.Resize((target_size[1], target_size[0]))(to_image((pca_rgb * 255).astype(np.uint8)))
    upscaled = np.asarray(upscaled_pil).astype(np.float32) / 255.0
    if upscaled.ndim == 2:
        upscaled = np.stack([upscaled] * 3, axis=-1)
    return np.clip(upscaled, 0.0, 1.0)


def visualize_flow_triplet(
    rgb: torch.Tensor,
    context: torch.Tensor,
    pred_rggb: torch.Tensor,
    gt_rggb: torch.Tensor,
    save_path: str,
    gamma: float,
    img_size: Union[int, Tuple[int, int]],
    patch_size: int,
    student_features: Optional[Dict[str, torch.Tensor]] = None,
    teacher_tokens: Optional[Dict[str, torch.Tensor]] = None,
    ae_tokens: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_rgb(img: torch.Tensor) -> np.ndarray:


        return tensor_to_numpy(img)

    def rggb_display(tensor: torch.Tensor) -> np.ndarray:


        return to_rgb(tensor[0])

    rgb_np = tensor_to_numpy(rgb[0])
    context_np = tensor_to_numpy(context[0])
    pred_np = rggb_display(pred_rggb)
    gt_np = rggb_display(gt_rggb)

    def visualize_tokens(tokens: torch.Tensor) -> np.ndarray:
        return dino_v2_visualize(tokens[:1].float(), img_size=img_size, patch_size=patch_size)

    student_images: List[np.ndarray] = []
    student_titles: List[str] = []
    if student_features is not None:
        student_order = [
            ("visible", "Student visible-light latent"),
            ("pred", "Student DiT predicted latent"),
            ("infrared", "Student infrared latent"),
        ]
        for key, title in student_order:
            tokens = student_features.get(key)
            if tokens is None:
                continue
            student_images.append(visualize_tokens(tokens))
            student_titles.append(title)

    teacher_images: List[np.ndarray] = []
    teacher_titles: List[str] = []
    if teacher_tokens is not None:
        teacher_order = [
            ("visible", "Teacher visible-light latent"),
            ("infrared", "Teacher infrared latent"),
        ]
        for key, title in teacher_order:
            tokens = teacher_tokens.get(key)
            if tokens is None:
                continue
            teacher_images.append(visualize_tokens(tokens))
            teacher_titles.append(title)

    def rggb_display_from_tensor(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach().cpu()
        return rggb_display(tensor)

    row1_images = [rgb_np, pred_np, gt_np]
    row1_titles = ["Visible-light input", "Generated infrared", "Target infrared"]
    if ae_tokens is not None:
        decoder_refined_tensor = ae_tokens.get("decoder_refined")
        if decoder_refined_tensor is not None:

            row1_images.append(rggb_display_from_tensor(decoder_refined_tensor[:1]))
            row1_titles.append("Decoder refined")
        shallow_tensor = ae_tokens.get("decoder_refined_pre_shallow")
        if shallow_tensor is not None:
            row1_images.append(rggb_display_from_tensor(shallow_tensor[:1]))
            row1_titles.append("Shallow refine")

    row_defs = [("images", row1_images, row1_titles)]
    if student_images:
        row_defs.append(("student", student_images, student_titles))
    if teacher_images:
        row_defs.append(("teacher", teacher_images, teacher_titles))

    num_rows = len(row_defs)
    cols = max(len(images) for _, images, _ in row_defs)

    fig, axes = plt.subplots(num_rows, cols, figsize=(3 * cols, 3 * num_rows))

    if num_rows == 1 and cols == 1:
         axes = np.array([axes])
    axes = np.array(axes, dtype=object).reshape(num_rows, -1)


    if len(axes.shape) == 1:
        axes = axes.reshape(num_rows, cols)

    for row_idx, (_, images, titles) in enumerate(row_defs):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(images):
                ax.imshow(images[col_idx])
                ax.set_title(titles[col_idx])
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)

import os, datetime, json, pathlib
import torch, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from metrics import AlignmentMetrics   # PRH repo

models = {
    "intern": {
        "text_only_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",
        "text_only_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_random_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",

    },
    "cog": {
        "text_only_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",
        "text_only_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_random_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",

    },
    "minicpm": {
        "text_only_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",
        "text_only_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_random_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",
    },
    "llava_7b": {
        "text_only_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_wit": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",
        "text_only_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_random_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img_hl": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",
    }
}

# ---------- PARAMETERS ----------
OUT_ROOT      = "./alignment_matrices"         # one folder for everything
METRICS       = [("cknna", 10), ("mutual_knn", 10)]
CMAP          = "viridis"

# ---------- UTILITIES ----------
def ensure_dir(*parts):
    path = pathlib.Path(os.path.join(*parts))
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalise_if(x, do=True):
    return F.normalize(x, p=2, dim=-1) if do else x


# ---------- MATRIX BUILDER ----------
@torch.no_grad()
def score_matrix(x_feats, y_feats, metric, topk=10, l2norm=True):
    # unpack tensor -> list[layer] for easier loop
    if isinstance(x_feats, torch.Tensor):
        x_layers = [x_feats[:, i, :] for i in range(x_feats.shape[1])]
    else:
        x_layers = x_feats
    if isinstance(y_feats, torch.Tensor):
        y_layers = [y_feats[:, j, :] for j in range(y_feats.shape[1])]
    else:
        y_layers = y_feats

    Lx, Ly = len(x_layers), len(y_layers)
    M = torch.zeros(Lx, Ly, device=x_layers[0].device)

    kwargs = {"topk": topk} if "knn" in metric else {}
    for i, xi in enumerate(x_layers):
        xi = normalise_if(xi, l2norm)
        for j, yj in enumerate(y_layers):
            yj = normalise_if(yj, l2norm)
            M[i, j] = AlignmentMetrics.measure(metric, xi, yj, **kwargs)

    return M.cpu().numpy()


# ---------- PLOTTING ----------
def save_heatmap(M, save_png, title="", vmin=0, vmax=1, cmap=CMAP):
    plt.figure(figsize=(6, 5))
    plt.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("layers Y"); plt.ylabel("layers X")
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.close()


# ---------- MAIN LOOP ----------
def compare_all(models, modality):
    now   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir   = ensure_dir(OUT_ROOT, modality + "_" + now)
    log_path  = out_dir / "log.txt"

    with open(log_path, "w") as LOG:
        LOG.write(f"Run started {now}\n")
        LOG.write(f"Modality = {modality}\n")
        LOG.write(f"Metrics  = {METRICS}\n\n")

        for (name_a, cfg_a), (name_b, cfg_b) in combinations(models.items(), 2):
            fpath_a = cfg_a[modality]
            fpath_b = cfg_b[modality]
            print(f"→ {name_a}  vs  {name_b}")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            feats_a = torch.load(fpath_a, map_location=DEVICE)  # adjust if the tensor is directly stored
            feats_b = torch.load(fpath_b, map_location=DEVICE)

            for metric, k in METRICS:
                M = score_matrix(feats_a, feats_b, metric, k)
                best = np.unravel_index(M.argmax(), M.shape)
                best_val = float(M[best])

                # ---------- save ----------
                tag = f"{name_a}_VS_{name_b}_{metric}"
                npy_path  = out_dir / f"{tag}.npy"
                png_path  = out_dir / f"{tag}.png"
                np.save(npy_path, M)
                save_heatmap(M, png_path,
                             title=f"{tag}\n(best={best_val:.4f} @ {best})")

                # ---------- log ----------
                LOG.write(json.dumps({
                    "pair":      f"{name_a} vs {name_b}",
                    "metric":    metric,
                    "k":         int(k),
                    "best_val":  float(best_val),
                    "layer_xy":  [int(best[0]), int(best[1])],
                    "matrix":    str(npy_path),
                    "figure":    str(png_path)
                }) + "\n")

            # tidy GPU RAM
            del feats_a, feats_b
            torch.cuda.empty_cache()

    print(f"\nFinished. Everything is in {out_dir}\n→ best values logged to {log_path}")


# --------- RUN ----------
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    compare_all(models, "text_only_wit")
    compare_all(models, "w_images_wit")
    compare_all(models, "w_random_img_wit")
    compare_all(models, "w_trina_img_wit")
    compare_all(models, "text_only_hl")
    compare_all(models, "w_random_img_hl")
    compare_all(models, "w_trina_img_hl")

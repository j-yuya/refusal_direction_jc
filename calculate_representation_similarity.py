import torch
from metrics import AlignmentMetrics
import platonic
from itertools import combinations
import numpy as np


models = {
    "intern": {
        "text_only": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",

    },
    "cog": {
        "text_only": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/cogvlm2/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",

    },
    "minicpm": {
        "text_only": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",

    },
    "llava_7b": {
        "text_only": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_text_only.pt",
        "w_images": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images.pt",
        "w_random_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_random.pt",
        "w_trina_img": "/work/jcaspary/refusal_direction_image_load/pipeline/runs/reproduction-llava-v15+7b/harmful_advbench_vlm_wit_1024/0.1_0/n_train_harmful_100/representations_w_images_trina.pt",

    }
}



def compare_features(model_dict, modality):
    print(f"\n--- Comparing {modality} features ---\n")
    for model_a, model_b in combinations(model_dict.keys(), 2):
        path_a = model_dict[model_a][modality]
        path_b = model_dict[model_b][modality]

        feature_a = torch.load(path_a)
        feature_b = torch.load(path_b)

        M = compute_score_matrix(feature_a, feature_b,
                         metric="cknna", topk=10, normalize=True)
        best = np.unravel_index(M.argmax(), M.shape)
        print("best score cknna =", M[best], "at layers", best)
        M = compute_score_matrix(feature_a, feature_b,
                         metric="mutual_knn", topk=10, normalize=True)
        best = np.unravel_index(M.argmax(), M.shape)
        print("best score mutual_knn =", M[best], "at layers", best)



compare_features(models, "text_only")
compare_features(models, "w_images")
compare_features(models, "w_random_img")
compare_features(models, "w_trina_img")


def compute_score_matrix(x_feats, y_feats,
                         metric="mutual_knn", topk=10, normalize=True):
    """
    Return the full Lx × Ly alignment matrix instead of only the best entry.

    Args
    ----
    x_feats : torch.Tensor (B, Lx, Dx) **or** list of length Lx with (B, Dx)
    y_feats : torch.Tensor (B, Ly, Dy) **or** list of length Ly with (B, Dy)
    metric  : see metrics.AlignmentMetrics
    topk    : k for *knn*-based metrics
    normalize : if True apply L2-normalisation to each feature vector

    Returns
    -------
    M       : np.ndarray shape (Lx, Ly) containing the alignment score
    """
    # ---------- unwrap tensors into lists of layers ----------
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

    # ---------- iterate ----------
    for i, x in enumerate(x_layers):
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        for j, y in enumerate(y_layers):
            if normalize:
                y = F.normalize(y, p=2, dim=-1)

            kwargs = {"topk": topk} if "knn" in metric else {}
            with torch.no_grad():      # saves a bit of memory
                score = AlignmentMetrics.measure(
                    metric, x, y, **kwargs
                )
            M[i, j] = score

    return M.cpu().numpy()            # convenient for saving/plotting

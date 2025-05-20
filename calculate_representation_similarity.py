import torch
from platonic import compute_score



path_1 = "/work/jcaspary/refusal_direction_image_load/pipeline/runs/6f6d72be3c7a8541d2942691c46fbd075c147352/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations.pt"
path_2 = "/work/jcaspary/refusal_direction_image_load/pipeline/runs/minicpm/harmful_advbench_vlm_harmless/0.1_0/n_train_harmful_100/representations.pt"

feature_1 = torch.load(path_1)
feature_2 = torch.load(path_2)

print(feature_1.shape)
print(feature_2.shape)

score = compute_score(feature_1, feature_2, 'cknna', topk=10, normalize=True)
print(score)



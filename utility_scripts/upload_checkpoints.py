import os
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
for d in sorted(os.listdir(".")):
    if not (d.startswith("checkpoints_") and os.path.isdir(d)):
        continue
    short = d.removeprefix("checkpoints_latentqa_cls_past_lens_").removeprefix("checkpoints_")
    name = f"sao-{short}".lower()
    assert len(name) <= 96, f"name too long: {name}"
    repo = f"{user}/{name}"
    create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=d, repo_type="model", allow_patterns=["final/*"])
    print("uploaded", repo)
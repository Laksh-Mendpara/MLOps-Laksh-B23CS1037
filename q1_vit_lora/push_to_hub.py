from huggingface_hub import HfApi
from config import Config

def push_model_to_hub(model_dir, repo_id):
    api = HfApi()
    token = Config.HF_TOKEN
    if not token:
        print("Warning: HF_Token not found, skipping HF upload.")
        return
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model", token=token)
    print("Successfully uploaded to HuggingFace!")

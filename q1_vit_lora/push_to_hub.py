from huggingface_hub import HfApi
from config import Config

def resolve_repo_id(api, token, repo_id):
    user_info = api.whoami(token=token)
    username = user_info["name"]
    org_names = {org["name"] for org in user_info.get("orgs", [])}

    if "/" not in repo_id:
        return f"{username}/{repo_id}"

    namespace, repo_name = repo_id.split("/", 1)
    if namespace == username or namespace in org_names:
        return repo_id

    print(
        f"Warning: namespace '{namespace}' is not accessible for the authenticated user '{username}'. "
        f"Uploading to '{username}/{repo_name}' instead."
    )
    return f"{username}/{repo_name}"

def push_model_to_hub(model_dir, repo_id):
    api = HfApi()
    token = Config.HF_TOKEN
    if not token:
        print("Warning: HF_Token not found, skipping HF upload.")
        return
    resolved_repo_id = resolve_repo_id(api, token, repo_id)
    print(f"Uploading model artifacts from '{model_dir}' to '{resolved_repo_id}'...")
    api.create_repo(repo_id=resolved_repo_id, token=token, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=resolved_repo_id, repo_type="model", token=token)
    print(f"Successfully uploaded to HuggingFace: https://huggingface.co/{resolved_repo_id}")
    return resolved_repo_id

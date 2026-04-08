from huggingface_hub import HfApi 
api = HfApi() 
api.create_repo(repo_id='RiyaShajan/ai-email-intelligence-environment', repo_type='dataset', exist_ok=True) 
api.upload_folder(folder_path='.', repo_id='RiyaShajan/ai-email-intelligence-environment', repo_type='dataset') 

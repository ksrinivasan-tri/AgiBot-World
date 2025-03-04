from huggingface_hub import HfApi

api = HfApi()
repo_id = "agibot-world/AgiBotWorld-Alpha"
repo_type = "dataset"

# Get metadata (file list, sizes, etc.)
repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type)

# repo_info.siblings is a list of files in the repo
# Each item has attributes rfilename, size, etc.
file_list = []
for sibling in repo_info.siblings:
    file_path = sibling.rfilename
    file_size = sibling.size if sibling.size is not None else 0
    file_list.append((file_path, file_size))

# Sort files by size or by path—whichever is helpful
file_list.sort(key=lambda x: x[1], reverse=True)

# Print them out or save to a CSV/JSON
for path, size in file_list:
    print(f"{path}")


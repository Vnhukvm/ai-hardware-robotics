快速下载download\_hf\_files.py

python3 download\_hf\_files.py IPEC-COMMUNITY/spatialvla-4b-224-pt main --repo-type model --download\_path /data1/spatialvla-4b-224-pt

python3 download\_hf\_files.py google/gemma-3-1b-it main --repo-type model --download\_path ./gemma-3-1b-it

python3 download\_hf\_files.py lerobot/smolvla_base main --repo-type model --download\_path ./smolvla_base

python3 download\_hf\_files.py nikriz/aopoli-lv-libero_combined_no_noops_lerobot_v21 main --repo-type dataset --download\_path /home/vipuser/217data/aopoli-lv-libero

python3 hf_downloader.py nikriz/aopoli-lv-libero_combined_no_noops_lerobot_v21 main --repo-type dataset --download\_path /home/vipuser/217data/aopoli-lv-libero-new

python3 hf_downloader.py openvla/modified_libero_rlds main --repo-type dataset --download_path ./modified_libero_rlds_data

python3 download\_hf\_files.py IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge main --repo-type model --download\_path ./spatialvla-4b-224-sft-bridge

python3 download\_hf\_files.py IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge main --repo-type model --download\_path ./spatialvla-4b-224-sft-bridge

`python3 download_hf_files_new.py IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge main --repo-type model --download_path ./spatialvla-4b-224-sft-bridge --generate-only`

`python3 download_hf_files_new.py IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge main --repo-type model --download_path ./spatialvla-4b-224-sft-bridge --download-only`

python3 download_hf_files_new.py IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal main --repo-type model --download_path ./spatialvla-4b-224-sft-fractal

python3 download_hf_files_new.py facebook/dinov3-vits16-pretrain-lvd1689m main --repo-type model --download_path ./dinov3-vits16-pretrain-lvd1689m

python hf_downloader.py yifengzhu-hf/LIBERO-datasets main --repo-type dataset --subfolder libero_spatial --download_path ./libero-dataset/dataset/ --download-only

![image.png](/./assets/432e92f2-4655-474d-8551-8b70d0f25541.png)

![image.png](/./assets/c669c2be-3b85-4aee-a1fd-c74df4106f9d.png)

需要切换到session所在路径，方可断点续传

aria2c -c -i download_links.txt

python hf_downloader.py yifengzhu-hf/LIBERO-datasets main --repo-type dataset --subfolder libero_spatial --download_path ./libero-dataset/dataset/ --generate-only

python hf_downloader.py yifengzhu-hf/LIBERO-datasets main --repo-type dataset --subfolder libero_10 --download_path ./libero-dataset/dataset/ --generate-only



python hf_downloader.py spatialverse/InteriorAgent main --repo-type dataset --download_path ./interiornav_data/scene_data --generate-only

python hf_downloader.py spatialverse/InteriorAgent main --repo-type dataset --download_path ./interiornav_data/scene_data --download-only

python hf_downloader.py nvidia/GR00T-N1.6-3B main --repo-type model --download_path ./GR00T-N1.6-3B



python hf_downloader.py shashuo0104/phystwin-toy main --repo-type model --download_path ./rope

python hf_downloader.py shashuo0104/phystwin-rope main --repo-type model --download_path ./sloth

python hf_downloader.py shashuo0104/phystwin-T-block main --repo-type model --download_path ./T


```python
import os
import requests
import json
import argparse
from urllib.parse import urljoin, quote
import re # 确保导入 re 模块

def fetch_file_list(repo_id, branch, repo_type, hf_token):
    """
    Fetches a recursive list of files from a Hugging Face repository,
    handling API pagination for very large repositories.
    repo_type can be 'model' or 'dataset'.
    """
    all_files = []
    next_url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/{branch}?recursive=true"
  
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    page_num = 1
    while next_url:
        print(f"Fetching file list page {page_num} from: {next_url}")
        response = requests.get(next_url, headers=headers)
  
        if response.status_code != 200:
            print(f"Failed to fetch data from {next_url}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            print("Please ensure your HF_TOKEN is correct and has the necessary permissions.")
            return None 

        all_files.extend(response.json())

        link_header = response.headers.get("Link")
        next_url = None 
        if link_header:
            match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            if match:
                next_url = match.group(1)
  
        page_num += 1

    if all_files is None:
        exit(1)

    return all_files

# --- MODIFIED FUNCTION ---
def generate_links_file(file_list, repo_id, branch, download_path, repo_type, subfolder=None): # <--- MODIFIED: Added subfolder parameter
    """
    Generates the download_links.txt file from the file list.
    Optionally filters for a specific subfolder.
    """
    os.makedirs(download_path, exist_ok=True)
  
    download_links_with_out = []
  
    if repo_type == 'dataset':
        base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/"
    else:
        base_url = f"https://huggingface.co/{repo_id}/resolve/{branch}/"
  
    # <--- MODIFIED: Filtering logic starts here
    if subfolder:
        # 规范化路径，确保它以'/'结尾，以便正确匹配前缀
        prefix = subfolder.strip('/') + '/'
        print(f"Filtering for files within subfolder: {prefix}")
        filtered_list = [f for f in file_list if f.get('type') == 'file' and f['path'].startswith(prefix)]
        print(f"Found {len(filtered_list)} files to download in the specified subfolder.")
    else:
        filtered_list = file_list
    # <--- MODIFIED: Filtering logic ends here

    for file_info in filtered_list:
        if file_info.get('type') == 'file':
            file_path = file_info['path']
  
            local_file_dir = os.path.dirname(os.path.join(download_path, file_path))
            if local_file_dir:
                os.makedirs(local_file_dir, exist_ok=True)
  
            file_url = urljoin(base_url, quote(file_path, safe=''))
            # aria2c 的 out 选项会保留相对路径，所以下载后会自动创建子文件夹
            aria2c_options = f"  out={file_path}" 
            download_links_with_out.append((file_url, aria2c_options))

    if not download_links_with_out:
        print("No files found to generate links for. Check your repo_id, branch, and subfolder path.")
        return

    download_file_path = os.path.join(download_path, "download_links.txt")
    with open(download_file_path, "w") as f:
        for url, options in download_links_with_out:
            f.write(f"{url}\n{options}\n")
      
    print(f"Successfully generated {len(download_links_with_out)} file links in {download_file_path}")

def start_download(download_path, hf_token):
    """
    Starts the download using aria2c with an existing download_links.txt file.
    """
    download_file_path = os.path.join(download_path, "download_links.txt")
    if not os.path.exists(download_file_path):
        print(f"Error: download_links.txt not found in {download_path}")
        print("Please generate the links file first using the --generate-only option.")
        exit(1)

    aria2_header_option = ""
    if hf_token:
        aria2_header_option = f'--header="Authorization: Bearer {hf_token}"'

    session_file_path = os.path.join(download_path, "aopoli.session")

    print("Starting download with aria2c (using session file for speed)...")
    command = f'aria2c -x 16 -c --retry-wait=5 --max-tries=0 {aria2_header_option} -i "{download_file_path}" -d "{download_path}" --save-session="{session_file_path}" --save-session-interval=60'

    exit_code = os.system(command)
    return exit_code

def mark_completion(download_path):
    """
    Appends a completion message to the download_links.txt file.
    """
    download_file_path = os.path.join(download_path, "download_links.txt")
    if os.path.exists(download_file_path):
        with open(download_file_path, "a") as f:
            f.write("\n# 全部下载完成\n")
        print("Marked download_links.txt as complete.")

def main():
    parser = argparse.ArgumentParser(description="Download files from Hugging Face repository using aria2c")
    parser.add_argument("repo_id", help="The Hugging Face repository ID")
    parser.add_argument("branch", help="The branch to download from")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"], help="Type of the repository. Default is 'model'.")
    parser.add_argument("--download_path", default=None, help="The path to save files. Defaults to a directory named after the repo_id.")
  
    # <--- MODIFIED: Added new argument
    parser.add_argument("--subfolder", default=None, help="Specify a subfolder to download. Only files within this folder will be downloaded.")

    parser.add_argument("--generate-only", action="store_true", help="Only generate the download_links.txt file and exit.")
    parser.add_argument("--download-only", action="store_true", help="Start download using an existing download_links.txt file.")
  
    args = parser.parse_args()
  
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token and not args.generate_only:
        print("Warning: HF_TOKEN environment variable is not set. Downloads for gated models may fail.")

    download_path = args.download_path
    if download_path is None:
        download_path = f"./{args.repo_id.split('/')[-1]}"

    if args.generate_only:
        print("Mode: Generate links only")
        file_list = fetch_file_list(args.repo_id, args.branch, args.repo_type, hf_token)
        if file_list:
            # <--- MODIFIED: Pass subfolder argument
            generate_links_file(file_list, args.repo_id, args.branch, download_path, args.repo_type, args.subfolder)
        else:
            print("Could not generate links file because fetching file list failed.")
      
    elif args.download_only:
        print("Mode: Download only")
        exit_code = start_download(download_path, hf_token)
        if exit_code == 0:
            print("aria2c completed successfully.")
            mark_completion(download_path)
        else:
            print(f"aria2c exited with error code: {exit_code}. Completion not marked.")

    else: # Default mode: generate and then download
        print("Mode: Generate and Download")
        file_list = fetch_file_list(args.repo_id, args.branch, args.repo_type, hf_token)
        if file_list:
            # <--- MODIFIED: Pass subfolder argument
            generate_links_file(file_list, args.repo_id, args.branch, download_path, args.repo_type, args.subfolder)
            exit_code = start_download(download_path, hf_token)
            if exit_code == 0:
                print("aria2c completed successfully.")
                mark_completion(download_path)
            else:
                print(f"aria2c exited with error code: {exit_code}. Completion not marked.")
        else:
            print("Could not start download because fetching file list failed.")

if __name__ == "__main__":
    main()
```

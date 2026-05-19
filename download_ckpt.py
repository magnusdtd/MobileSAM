from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="magnusdtd/ML-Final-Project-MobileSAM", local_dir="outputs/weights", local_dir_use_symlinks=False
)

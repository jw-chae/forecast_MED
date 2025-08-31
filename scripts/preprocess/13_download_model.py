from huggingface_hub import snapshot_download

snapshot_download(
    repo_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    local_dir = "/home/joongwon00/Project_Tsinghua_Paper/med_deepseek/models",
    #allow_patterns = ["*UD-IQ1_S*"],  # Download only 1.58-bit variant
)
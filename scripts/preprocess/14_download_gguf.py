from huggingface_hub import snapshot_download

# Download specific GGUF version (Q4_K_M is good balance of quality/size)
snapshot_download(
    repo_id="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
    local_dir="/home/joongwon00/Project_Tsinghua_Paper/med_deepseek/models_gguf_single",
    allow_patterns=["*Q4_K_M.gguf"],  # Only download Q4_K_M version
)
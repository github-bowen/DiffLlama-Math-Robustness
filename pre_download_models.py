from huggingface_hub import snapshot_download

# Download checkpoint-64434 for DiffLlama-375M model
diffllama_path = snapshot_download(
    repo_id="reyllama/DiffLlama-375M",
    cache_dir="./cache",
    allow_patterns=["checkpoint-64434/*"],
    force_download=False
)

# Download checkpoint-64434 for Llama_375M model
llama_path = snapshot_download(
    repo_id="reyllama/Llama_375M",
    cache_dir="./cache",
    allow_patterns=["checkpoint-64434/*"],
    force_download=False
)

print(f"DiffLlama-375M checkpoint-64434 downloaded to: {diffllama_path}")
print(f"Llama_375M checkpoint-64434 downloaded to: {llama_path}")

"""
DiffLlama-375M checkpoint-64434 downloaded to: ./cache/models--reyllama--DiffLlama-375M/snapshots/8960f22033190f1560537f4932fe649828ef53e2
Llama_375M checkpoint-64434 downloaded to: ./cache/models--reyllama--Llama_375M/snapshots/416b70824d560b02245268c208ffd5388b4aa056
"""
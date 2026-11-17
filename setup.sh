#!/usr/bin/env bash
set -euo pipefail          # stop at the first error

# ---- 0. config ----
REPO=https://github.com/deepanshi-s/mae_reproduction.git
WORK=/workspace
CODE=$WORK/mae_reproduction                          
DATA=$WORK/data/imagenet100

# ---- 1. system tools ----
apt-get update -qq && apt-get install -y -qq tmux htop rsync git

# ---- 2. code ----
cd $WORK
[ -d mae_reproduction ] || git clone $REPO      # skip if already cloned
cd $CODE && git pull

# ---- 3. python deps ----
pip install -q -r requirements.txt

# ---- 4. secrets (never hardcoded) ----
: "${WANDB_API_KEY:?set WANDB_API_KEY first}"
wandb login "$WANDB_API_KEY"

# ---- 5. data: download if missing, then split into train/ and val/ ----
if [ ! -d "$DATA/data/train" ]; then                 
  pip install -q huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download('clane9/imagenet-100', repo_type='dataset', local_dir='$DATA')"   : no CLI/PATH issues

  cd "$DATA/data"                                    
  mkdir -p train val                                 
  mv train-*.parquet train/ 2>/dev/null || true      
  mv validation-*.parquet val/ 2>/dev/null || true   
fi
echo "train shards: $(ls $DATA/data/train | wc -l)  val shards: $(ls $DATA/data/val | wc -l)"   : expect 17 / 1

# ---- 6. sanity + smoke test ----
cd $CODE                                             
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
#python main.py --wand_id test_session1                         
echo "✅ setup done"
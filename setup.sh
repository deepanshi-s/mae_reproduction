#!/usr/bin/env bash
set -euo pipefail          # stop at the first error

# ---- 0. config ----
REPO=https://github.com/deepanshi-s/mae_reproduction.git
WORK=/workspace
DATA=$WORK/data/imagenet100

# ---- 1. system tools ----
apt-get update -qq && apt-get install -y -qq tmux htop rsync git

# ---- 2. code ----
cd $WORK
[ -d mae_reproduction ] || git clone $REPO      # skip if already cloned
cd mae_reproduction && git pull

# ---- 3. python deps ----
pip install -q -r requirements.txt

# ---- 4. secrets (never hardcoded) ----
: "${WANDB_API_KEY:?set WANDB_API_KEY first}"
wandb login "$WANDB_API_KEY"

# ---- 5. data (download only if missing) ----
if [ ! -d "$DATA" ]; then
  pip install -q huggingface_hub
  hf download clane9/imagenet-100 --repo-type dataset --local-dir $DATA
fi

# ---- 6. sanity + smoke test ----
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
echo "✅ setup done"
#!/bin/bash
# run_and_push.sh - Run PINN benchmark with time limit and auto-push latest results to GitHub

set -e
export PYTHONUNBUFFERED=1

# Default values
SCALE="large"
PRECISION="float32"
BATCH_SIZE=50000
GPUS="0,1,2,3"
EPOCHS=15000
TIME_LIMIT=13200  # Default 3 hours 40 minutes (13200 seconds)
RESUME=""
TOKEN=""
PORT=29500
OUT_DIR=""
PROXY=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scale) SCALE="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --time_limit) TIME_LIMIT="$2"; shift ;;
        --resume) RESUME="$2"; shift ;;
        --token) TOKEN="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --out_dir) OUT_DIR="$2"; shift ;;
        --proxy) PROXY="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️ Warning: Neither --token nor GITHUB_TOKEN environment variable is set."
    echo "Automatic push might fail if credentials are not configured on the server."
fi

# Use the token from argument or environment variable
PUSH_TOKEN=${TOKEN:-$GITHUB_TOKEN}

# Calculate number of GPUs
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT_DIR=${OUT_DIR:-"outputs_${SCALE}_${PRECISION}_${NUM_GPUS}gpus"}

echo "============================================="
echo "🚀 Starting Scheduled PINN Run with Time Limit"
echo "  - Scale: $SCALE"
echo "  - Precision: $PRECISION"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Visible GPUs: $GPUS (Count: $NUM_GPUS)"
echo "  - Master Port: $PORT"
echo "  - Time Limit: $TIME_LIMIT seconds (~$(echo "scale=2; $TIME_LIMIT/3600" | bc) hours)"
echo "  - Output Directory: $OUT_DIR"
echo "============================================="

echo "📦 Upgrading pip..."
pip install --upgrade pip
echo "📦 Installing/checking dependencies..."
pip install -r requirements_linux_rocm.txt

export HIP_VISIBLE_DEVICES=$GPUS
export CUDA_VISIBLE_DEVICES=$GPUS

# Build main command arguments
CMD_ARGS="--scale $SCALE --precision $PRECISION --batch_size $BATCH_SIZE --epochs $EPOCHS --time_limit $TIME_LIMIT --out_dir $OUT_DIR"
if [ -n "$RESUME" ]; then
    CMD_ARGS="$CMD_ARGS --resume $RESUME"
fi

# Execute python training
if [ $NUM_GPUS -gt 1 ]; then
    echo "🔥 Running DDP Multi-GPU Mode..."
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT main.py $CMD_ARGS
else
    echo "🔥 Running Single GPU Mode..."
    python3 main.py $CMD_ARGS
fi

echo "============================================="
echo "🧹 Preparing files for Git upload..."

# Find the latest checkpoint and delete all other older checkpoints to save space
CKPT_DIR="${OUT_DIR}/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    # Find the latest checkpoint file by modification time
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Latest checkpoint found: $LATEST_CKPT"
        # Delete all other checkpoints in this folder
        for f in "$CKPT_DIR"/*.pt; do
            if [ "$f" != "$LATEST_CKPT" ]; then
                rm -f "$f"
            fi
        done
        echo "Cleaned up older checkpoints. Kept only the latest one."
    else
        echo "No checkpoints found to clean."
    fi
fi

# Configure local git user if not set, to prevent commit failure
if ! git config user.email >/dev/null 2>&1; then
    git config user.email "benchmark-bot@example.com"
fi
if ! git config user.name >/dev/null 2>&1; then
    git config user.name "Benchmark Bot"
fi

# Stage files for git
# Generate hardware profiling graphs before pushing
if [ -d "$OUT_DIR/profiling" ]; then
    echo "📊 Generating hardware profiling graphs..."
    python3 analyze_hardware.py --dir "$OUT_DIR" || true
fi

# Using flock to ensure multiple concurrent scripts don't conflict during git operations
(
  flock -n 200 || { echo "🔒 Waiting for other PINN tests to finish Git operations..."; flock 200; }

  # Force add profiling logs and generated graphs (since outputs/profiling/ is in .gitignore)
  git add -f "$OUT_DIR/figures/" "$OUT_DIR/profiling/"
  if [ -d "$CKPT_DIR" ]; then
      # Force add because *.pt is in .gitignore
      git add -f "$CKPT_DIR"
  fi

  # Commit changes
  COMMIT_MSG="chore: auto-save benchmark results scale=${SCALE} precision=${PRECISION} gpus=${NUM_GPUS} date=$(date +'%Y-%m-%d %H:%M:%S')"
  git commit -m "$COMMIT_MSG" || echo "No changes to commit."

  # Push to GitHub
  BRANCH=$(git symbolic-ref --short -q HEAD)

  # Stash any local modifications to tracked files (like chmod +x changes or script micro-edits)
  # to guarantee a clean workspace for git pull --rebase
  echo "📦 Stashing any local changes/filemode changes..."
  git stash -q || true

  if [ -n "$PROXY" ]; then
      echo "🌐 Using Proxy: $PROXY for Git operations"
      export http_proxy=$PROXY
      export https_proxy=$PROXY
      export HTTP_PROXY=$PROXY
      export HTTPS_PROXY=$PROXY
      export all_proxy=$PROXY
  fi

  MAX_RETRIES=5
  RETRY_COUNT=0
  PUSH_SUCCESS=0

  if [ -n "$PUSH_TOKEN" ]; then
      echo "🚀 Pushing changes to GitHub using Personal Access Token..."
  else
      echo "⚠️ GITHUB_TOKEN not set. Attempting standard git push..."
  fi

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
      echo "🔄 Attempt $((RETRY_COUNT+1))/$MAX_RETRIES for Git Pull & Push..."
      
      if [ -n "$PUSH_TOKEN" ]; then
          # 先进行 pull --rebase，防止并行测试推送时产生的 Non-fast-forward 冲突
          git pull --rebase "https://${PUSH_TOKEN}@github.com/lockezhan/Antigravity-Project.git" "$BRANCH" || echo "Rebase skipped or no remote changes."
          # Push using Token-embedded HTTPS URL (去掉了 --force，防止覆盖其他并行进程的提交)
          if git push "https://${PUSH_TOKEN}@github.com/lockezhan/Antigravity-Project.git" "$BRANCH" -u; then
              PUSH_SUCCESS=1
              break
          fi
      else
          git pull --rebase origin "$BRANCH" || echo "Rebase skipped."
          if git push origin "$BRANCH" -u; then
              PUSH_SUCCESS=1
              break
          fi
      fi
      
      echo "❌ Push failed, network might be unstable. Retrying in 15 seconds..."
      sleep 15
      RETRY_COUNT=$((RETRY_COUNT+1))
  done

  if [ $PUSH_SUCCESS -eq 1 ]; then
      echo "✅ Push successful!"
  else
      echo "❌ Push failed after $MAX_RETRIES attempts."
  fi

  # Restore stashed local changes
  echo "📦 Restoring local changes..."
  git stash pop -q || true

) 200>git_push.lock

echo "============================================="
echo "🎉 Scheduled Run and Push complete!"
echo "============================================="

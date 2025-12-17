#!/usr/bin/env bash
set -euo pipefail

# Copy this file to `run_sbatch.sh` and fill in your own Slurm settings:
#   cp baselines/str_builder/run_sbatch.example.sh baselines/str_builder/run_sbatch.sh
#   chmod +x baselines/str_builder/run_sbatch.sh
#   vim baselines/str_builder/run_sbatch.sh
#
# Then submit:
#   ./baselines/str_builder/run_sbatch.sh baselines/str_builder/config.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
if command -v realpath >/dev/null 2>&1; then
  BASE_CONFIG_PATH="$(realpath "${BASE_CONFIG_PATH}")"
fi

# Slurm resources (set these for your cluster)
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NTASKS="${NTASKS:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEM="${MEM:-100g}"
TIME="${TIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-llm_collab_mc_mc}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCRIPT_DIR}/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

# Conda env
CONDA_ENV="${CONDA_ENV:-comlrl}"
BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"

# Minecraft server (must exist on the compute node filesystem, e.g. shared $HOME)
MC_DIR="${MC_DIR:-$HOME/mc-server}"
MC_JAR="${MC_JAR:-paper-1.19.2-307.jar}"
MC_PORT="${MC_PORT:-25565}"
BOT_USERNAME="${BOT_USERNAME:-executor_bot}"
BOT_USERNAME2="${BOT_USERNAME2:-${BOT_USERNAME}_2}"
MC_XMS="${MC_XMS:-1G}"
MC_XMX="${MC_XMX:-1G}"

# Baseline args
EXTRA_ARGS="${EXTRA_ARGS:-}" # e.g. EXTRA_ARGS="--limit 1"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-auto}" # auto | jsonl | json

if [[ -z "${ACCOUNT}" || -z "${PARTITION}" || "${ACCOUNT}" == "..." || "${PARTITION}" == "..." || "${ACCOUNT}" == "YOUR_ACCOUNT" || "${PARTITION}" == "YOUR_PARTITION" ]]; then
  echo "ERROR: set ACCOUNT and PARTITION (env vars) for sbatch." >&2
  echo "Example:" >&2
  echo "  ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./baselines/str_builder/run_sbatch.sh baselines/str_builder/config.yaml" >&2
  exit 2
fi

echo "Submitting Slurm job (server + baseline in one job):"
echo "  base config: ${BASE_CONFIG_PATH}"
echo "  job name:    ${JOB_NAME}"
echo "  logs:        ${SLURM_LOG_DIR}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes="${NODES}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --ntasks="${NTASKS}" \
  --ntasks-per-node="${NTASKS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --job-name="${JOB_NAME}" \
  --output="${SLURM_LOG_DIR}/%x-%j.out" \
  --export=ALL,REPO_DIR="${REPO_DIR}",BASE_CONFIG_PATH="${BASE_CONFIG_PATH}",CONDA_ENV="${CONDA_ENV}",BASHRC_PATH="${BASHRC_PATH}",MC_DIR="${MC_DIR}",MC_JAR="${MC_JAR}",MC_PORT="${MC_PORT}",BOT_USERNAME="${BOT_USERNAME}",BOT_USERNAME2="${BOT_USERNAME2}",MC_XMS="${MC_XMS}",MC_XMX="${MC_XMX}",EXTRA_ARGS="${EXTRA_ARGS}",OUTPUT_FORMAT="${OUTPUT_FORMAT}" \
  <<'SBATCH'
#!/usr/bin/env bash
set -eo pipefail

echo "job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname)"
echo "repo=${REPO_DIR}"
echo "base_config=${BASE_CONFIG_PATH}"

if [ -f "${BASHRC_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${BASHRC_PATH}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found (set BASHRC_PATH or load conda module)" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node not found in PATH" >&2
  exit 1
fi

if [ ! -d "${REPO_DIR}/node_modules" ]; then
  echo "ERROR: ${REPO_DIR}/node_modules not found. Run once on login node:" >&2
  echo "  cd ${REPO_DIR} && npm install" >&2
  exit 1
fi

JAVA_BIN=""
if [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
  JAVA_BIN="${JAVA_HOME}/bin/java"
else
  JAVA_BIN="$(command -v java || true)"
fi
if [ -z "${JAVA_BIN}" ]; then
  echo "ERROR: java not found in PATH (set JAVA_HOME or load java module)" >&2
  exit 1
fi

if [ ! -d "${MC_DIR}" ]; then
  echo "ERROR: MC_DIR not found: ${MC_DIR}" >&2
  exit 1
fi
if [ ! -f "${MC_DIR}/${MC_JAR}" ]; then
  echo "ERROR: Minecraft server jar not found: ${MC_DIR}/${MC_JAR}" >&2
  exit 1
fi

mkdir -p "${MC_DIR}/logs"
if [ ! -f "${MC_DIR}/eula.txt" ]; then
  echo "eula=true" > "${MC_DIR}/eula.txt"
fi
if [ ! -f "${MC_DIR}/server.properties" ]; then
  echo "online-mode=false" > "${MC_DIR}/server.properties"
else
  sed -i -E "s/^online-mode=.*/online-mode=false/" "${MC_DIR}/server.properties" || true
fi

OPS_JSON="${MC_DIR}/ops.json" MC_USERNAMES="${BOT_USERNAME},${BOT_USERNAME2}" python3 - <<'PY'
import os, json, hashlib, uuid
ops_path = os.environ["OPS_JSON"]
usernames = [u.strip() for u in os.environ.get("MC_USERNAMES","").split(",") if u.strip()]

data = []
if os.path.exists(ops_path):
    try:
        data = json.load(open(ops_path, "r", encoding="utf-8"))
    except Exception:
        data = []
if not isinstance(data, list):
    data = []

def offline_uuid(name: str) -> str:
    s = ("OfflinePlayer:" + name).encode("utf-8")
    d = bytearray(hashlib.md5(s).digest())
    d[6] = (d[6] & 0x0F) | 0x30
    d[8] = (d[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(d)))

for username in usernames:
    uid = offline_uuid(username)
    found = False
    for e in data:
        if isinstance(e, dict) and e.get("name") == username:
            e["uuid"] = uid
            e["level"] = 4
            e.setdefault("bypassesPlayerLimit", False)
            found = True
    if not found:
        data.append({"uuid": uid, "name": username, "level": 4, "bypassesPlayerLimit": False})
    print("ops.json updated:", username, uid)

json.dump(data, open(ops_path, "w", encoding="utf-8"), indent=2)
PY

SERVER_LOG="${MC_DIR}/logs/server_${SLURM_JOB_ID:-$$}.log"
cd "${MC_DIR}"
"${JAVA_BIN}" -Xms"${MC_XMS}" -Xmx"${MC_XMX}" -jar "${MC_JAR}" --nogui < /dev/null > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "server_pid=${SERVER_PID}"
echo "server_log=${SERVER_LOG}"

cleanup() {
  echo "stopping server..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for server to start..."
for i in $(seq 1 60); do
  if grep -q "Done (" "${SERVER_LOG}" 2>/dev/null; then
    echo "server ready"
    break
  fi
  sleep 1
done

cd "${REPO_DIR}"
conda activate "${CONDA_ENV}"

TMP_CONFIG="${SLURM_TMPDIR:-/tmp}/str_builder_config_${SLURM_JOB_ID:-$$}.yaml"
export TMP_CONFIG
cp "${BASE_CONFIG_PATH}" "${TMP_CONFIG}"

python3 - <<'PY'
import os, yaml, pathlib
cfg_path = pathlib.Path(os.environ["TMP_CONFIG"])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

repo_dir = pathlib.Path(os.environ["REPO_DIR"])
job_id = os.environ.get("SLURM_JOB_ID") or "local"
job_name = os.environ.get("SLURM_JOB_NAME") or "llm_collab_mc_mc"
out_fmt = os.environ.get("OUTPUT_FORMAT","auto")

cfg.setdefault("dataset", {})
cfg["dataset"]["csv_path"] = str(repo_dir / "dataset/str_builder/data.csv")

cfg.setdefault("output", {})
base = cfg.get("output", {}).get("path", "outputs/out.jsonl")
suffix = pathlib.Path(str(base)).suffix.lower()
if out_fmt == "jsonl" or (out_fmt == "auto" and suffix == ".jsonl"):
    out_name = f"{job_name}-{job_id}.jsonl"
else:
    out_name = f"{job_name}-{job_id}.json"
cfg["output"]["path"] = str(repo_dir / "baselines/str_builder/outputs" / out_name)
cfg["output"]["overwrite"] = True

cfg.setdefault("minecraft", {})
cfg["minecraft"]["enabled"] = True
cfg["minecraft"]["host"] = "127.0.0.1"
cfg["minecraft"]["port"] = int(os.environ.get("MC_PORT","25565"))
cfg["minecraft"]["username"] = os.environ.get("BOT_USERNAME","executor_bot")
cfg["minecraft"]["username2"] = os.environ.get("BOT_USERNAME2", cfg["minecraft"]["username"] + "_2")
cfg["minecraft"]["origin_mode"] = "spawn_offset"

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("wrote tmp config:", cfg_path)
PY

python3 -u baselines/str_builder/main.py --config "${TMP_CONFIG}" ${EXTRA_ARGS}
SBATCH

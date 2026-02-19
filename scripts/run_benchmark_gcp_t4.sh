#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_benchmark_gcp_t4.sh [options]

Runs benchmark task on a Google Cloud VM (n1-standard-4 + T4), then downloads results.

Options:
  --project <id>            GCP project ID (default: gcloud active project)
  --zone <zone>             GCP zone (default: us-central1-a)
  --instance <name>         VM name (default: llama3-benchmark-t4)
  --task <gradle-task>      Gradle benchmark task (default: benchmarkAll)
  --repo <url>              Git repository URL (default: git remote origin)
  --branch <name>           Git branch to benchmark (default: current local branch)
  --image-project <name>    Boot image project (default: ubuntu-os-cloud)
  --image-family <name>     Boot image family (default: ubuntu-2204-lts)
  --disk-gb <size>          Boot disk size in GB (default: 200)
  --workdir <path>          Remote workdir (default: ~/llama3-java-hat)
  --output-dir <path>       Local output dir for benchmark files (default: build/benchmark-results/gcp)
  --no-download             Skip downloading result files
  --keep-instance           Do not stop instance after benchmark
  --destroy-instance        Delete instance after benchmark (full teardown)
  --destroy-only            Delete instance and exit (no benchmark run)
  --skip-create             Reuse existing instance; fail if it does not exist

Environment knobs (optional):
  RUN_OPENCL_BENCHMARKS     default: true
  BENCHMARK_ITERS           optional override for benchmark iteration count
  BENCHMARK_WARMUP_ITERS    optional override for benchmark warmup iteration count

Examples:
  scripts/run_benchmark_gcp_t4.sh --project my-proj --zone us-central1-a
  scripts/run_benchmark_gcp_t4.sh --task benchmarkInference --keep-instance
  scripts/run_benchmark_gcp_t4.sh --destroy-only
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

PROJECT=""
ZONE="us-central1-a"
INSTANCE="llama3-benchmark-t4"
TASK="benchmarkAll"
REPO_URL=""
BRANCH=""
IMAGE_PROJECT="ubuntu-os-cloud"
IMAGE_FAMILY="ubuntu-2404-lts"
DISK_GB="200"
WORKDIR="~/llama3-java-hat"
OUTPUT_DIR="build/benchmark-results/gcp"
KEEP_INSTANCE="false"
DESTROY_INSTANCE="false"
DESTROY_ONLY="false"
DOWNLOAD_RESULTS="true"
SKIP_CREATE="false"

RUN_OPENCL_BENCHMARKS="${RUN_OPENCL_BENCHMARKS:-true}"
BENCHMARK_ITERS="${BENCHMARK_ITERS:-}"
BENCHMARK_WARMUP_ITERS="${BENCHMARK_WARMUP_ITERS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --instance) INSTANCE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --image-project) IMAGE_PROJECT="$2"; shift 2 ;;
    --image-family) IMAGE_FAMILY="$2"; shift 2 ;;
    --disk-gb) DISK_GB="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --no-download) DOWNLOAD_RESULTS="false"; shift ;;
    --keep-instance) KEEP_INSTANCE="true"; shift ;;
    --destroy-instance) DESTROY_INSTANCE="true"; shift ;;
    --destroy-only) DESTROY_ONLY="true"; shift ;;
    --skip-create) SKIP_CREATE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

require_cmd gcloud
require_cmd git

if [[ "${KEEP_INSTANCE}" == "true" && "${DESTROY_INSTANCE}" == "true" ]]; then
  echo "Use only one of --keep-instance or --destroy-instance." >&2
  exit 1
fi

if [[ -z "${PROJECT}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "GCP project is not set. Pass --project or run: gcloud config set project <id>" >&2
  exit 1
fi

if [[ -z "${REPO_URL}" ]]; then
  REPO_URL="$(git config --get remote.origin.url || true)"
fi
if [[ -z "${REPO_URL}" ]]; then
  echo "Cannot infer repo URL from git remote. Pass --repo <url>." >&2
  exit 1
fi

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi
if [[ "${BRANCH}" == "HEAD" ]]; then
  echo "Detached HEAD detected. Pass --branch <name>." >&2
  exit 1
fi

echo "Project: ${PROJECT}"
echo "Zone: ${ZONE}"
echo "Instance: ${INSTANCE}"
echo "Machine: n1-standard-4 + nvidia-tesla-t4"
echo "Repo: ${REPO_URL}"
echo "Branch: ${BRANCH}"
echo "Task: ${TASK}"

if [[ "${DESTROY_ONLY}" == "true" ]]; then
  echo "Destroy-only mode: deleting instance ${INSTANCE} (if it exists)."
  if gcloud compute instances describe "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" >/dev/null 2>&1; then
    gcloud compute instances delete "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --quiet
  else
    echo "Instance ${INSTANCE} does not exist."
  fi
  exit 0
fi

if gcloud compute instances describe "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" >/dev/null 2>&1; then
  echo "Instance exists: ${INSTANCE}"
  if [[ "${DESTROY_INSTANCE}" == "true" ]]; then
    echo "Destroy mode enabled; recreating instance for a clean benchmark environment."
    gcloud compute instances delete "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --quiet >/dev/null
  else
    gcloud compute instances start "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" >/dev/null
  fi
fi

if ! gcloud compute instances describe "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" >/dev/null 2>&1; then
  if [[ "${SKIP_CREATE}" == "true" ]]; then
    echo "Instance ${INSTANCE} does not exist and --skip-create was passed." >&2
    exit 1
  fi

  echo "Creating instance ${INSTANCE}..."
  gcloud compute instances create "${INSTANCE}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "n1-standard-4" \
    --accelerator "type=nvidia-tesla-t4,count=1" \
    --maintenance-policy "TERMINATE" \
    --provisioning-model "STANDARD" \
    --create-disk "auto-delete=yes,boot=yes,device-name=${INSTANCE},image-family=${IMAGE_FAMILY},image-project=${IMAGE_PROJECT},mode=rw,size=${DISK_GB},type=pd-ssd" \
    --metadata "install-nvidia-driver=True" \
    --scopes "https://www.googleapis.com/auth/cloud-platform"
fi

echo "Waiting for SSH..."
for _ in {1..30}; do
  if gcloud compute ssh "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" --command "echo ssh-ok" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

REMOTE_SCRIPT="$(mktemp)"
cleanup_local() {
  rm -f "${REMOTE_SCRIPT}"
}
trap cleanup_local EXIT

cat > "${REMOTE_SCRIPT}" <<'EOF_REMOTE'
set -euo pipefail

REPO_URL="$1"
BRANCH="$2"
TASK="$3"
RUN_OPENCL_BENCHMARKS="$4"
BENCHMARK_ITERS="$5"
BENCHMARK_WARMUP_ITERS="$6"
WORKDIR="$7"

WORKDIR="${WORKDIR/#\~/$HOME}"
mkdir -p "$WORKDIR"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl build-essential autoconf cmake pkg-config ninja-build unzip zip \
  libx11-dev libxext-dev libxrender-dev libxrandr-dev libxtst-dev libxt-dev \
  libcups2-dev libasound2-dev libfontconfig1-dev clinfo ocl-icd-opencl-dev \
  ubuntu-drivers-common pciutils

echo "Checking NVIDIA driver availability..."
wait_for_nvidia() {
  local attempts="$1"
  local i
  for ((i=1; i<=attempts; i++)); do
    if nvidia-smi >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

# First wait: metadata-driven install on GCE can take several minutes.
if ! wait_for_nvidia 60; then
  echo "NVIDIA driver is not ready after waiting 10 minutes. Trying fallback install..."

  if ! lspci | grep -qi "NVIDIA"; then
    echo "NVIDIA GPU not detected by lspci; cannot continue." >&2
    exit 1
  fi

  sudo DEBIAN_FRONTEND=noninteractive ubuntu-drivers autoinstall || true
  sudo modprobe nvidia || true

  if ! wait_for_nvidia 60; then
    echo "NVIDIA driver is still not ready after fallback install." >&2
    ubuntu-drivers list || true
    lsmod | grep -i nvidia || true
    exit 1
  fi
fi

nvidia-smi
clinfo | sed -n '1,80p' || true

if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "Ensuring Boot JDK 25 is available..."
need_boot_jdk="true"
if command -v javac >/dev/null 2>&1; then
  if javac -version 2>&1 | grep -q "25"; then
    need_boot_jdk="false"
  fi
fi

if [[ "${need_boot_jdk}" == "true" ]]; then
  BOOT_JDK_DIR="$HOME/boot-jdk-25"
  if [[ ! -x "$BOOT_JDK_DIR/bin/javac" ]]; then
    TMP_BOOT_JDK_TGZ="$(mktemp)"
    curl -fsSL -o "$TMP_BOOT_JDK_TGZ" \
      "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse"
    rm -rf "$BOOT_JDK_DIR"
    mkdir -p "$BOOT_JDK_DIR"
    tar -xzf "$TMP_BOOT_JDK_TGZ" -C "$BOOT_JDK_DIR" --strip-components=1
    rm -f "$TMP_BOOT_JDK_TGZ"
  fi
  export JAVA_HOME="$BOOT_JDK_DIR"
  export PATH="$JAVA_HOME/bin:$PATH"
fi

if [[ ! -d "$HOME/babylon-jdk" ]]; then
  git clone --depth 1 --branch code-reflection https://github.com/openjdk/babylon "$HOME/babylon-jdk"
fi

cd "$HOME/babylon-jdk"
if [[ ! -x "build/linux-x86_64-server-release/images/jdk/bin/java" ]]; then
  bash configure --with-conf-name=linux-x86_64-server-release --with-debug-level=release --with-native-debug-symbols=none --disable-warnings-as-errors
  make images JOBS="$(nproc)" CONF=linux-x86_64-server-release
fi

if [[ ! -f "hat/build/hat-core-1.0.jar" ]]; then
  cd hat
  export JAVA_HOME="$HOME/babylon-jdk/build/linux-x86_64-server-release/images/jdk"
  export PATH="$JAVA_HOME/bin:$PATH"
  if ! bash hat/bootstrap.bash; then
    echo "HAT bootstrap failed. Dumping CMake logs for diagnosis..."
    find "$HOME/babylon-jdk/hat" -type f \( -name CMakeError.log -o -name CMakeOutput.log \) -print -exec tail -n 200 {} \;
    exit 1
  fi
  java -cp hat/job.jar --enable-preview --source 26 hat.java bld
fi

cd "$WORKDIR"
export JAVA_BABYLON_ROOT="$HOME/babylon-jdk"
export JAVA_HOME="$JAVA_BABYLON_ROOT/build/linux-x86_64-server-release/images/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
export LLAMA_FP16_PATH="$WORKDIR/Llama-3.2-1B-Instruct-f16.gguf"
export RUN_OPENCL_BENCHMARKS="$RUN_OPENCL_BENCHMARKS"

if [[ ! -f "$LLAMA_FP16_PATH" ]]; then
  chmod +x scripts/download_llama_fp16.sh
  ./scripts/download_llama_fp16.sh
fi

if [[ -n "$BENCHMARK_ITERS" ]]; then
  export BENCHMARK_ITERS
fi
if [[ -n "$BENCHMARK_WARMUP_ITERS" ]]; then
  export BENCHMARK_WARMUP_ITERS
fi

./gradlew clean "$TASK" --console=plain

mkdir -p "$WORKDIR/build/benchmark-results"
tar -C "$WORKDIR/build" -czf "$WORKDIR/build/benchmark-results/gcp-benchmark-artifacts.tgz" \
  benchmark-results reports/tests test-results || true
EOF_REMOTE

echo "Running benchmark on remote instance..."
REMOTE_SCRIPT_NAME="run-benchmark-${RANDOM}-$$.sh"
REMOTE_SCRIPT_PATH="~/${REMOTE_SCRIPT_NAME}"
gcloud compute scp "${REMOTE_SCRIPT}" "${INSTANCE}:${REMOTE_SCRIPT_PATH}" --project "${PROJECT}" --zone "${ZONE}" >/dev/null
gcloud compute ssh "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" \
  --command "bash ${REMOTE_SCRIPT_PATH} '$REPO_URL' '$BRANCH' '$TASK' '$RUN_OPENCL_BENCHMARKS' '$BENCHMARK_ITERS' '$BENCHMARK_WARMUP_ITERS' '$WORKDIR'; rc=\$?; rm -f ${REMOTE_SCRIPT_PATH}; exit \$rc"

mkdir -p "${OUTPUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "${DOWNLOAD_RESULTS}" == "true" ]]; then
  echo "Downloading benchmark TSV and artifacts..."
  gcloud compute scp \
    "${INSTANCE}:${WORKDIR}/build/benchmark-results/results.tsv" \
    "${OUTPUT_DIR}/results-${INSTANCE}-${STAMP}.tsv" \
    --project "${PROJECT}" \
    --zone "${ZONE}" || true

  gcloud compute scp \
    "${INSTANCE}:${WORKDIR}/build/benchmark-results/gcp-benchmark-artifacts.tgz" \
    "${OUTPUT_DIR}/artifacts-${INSTANCE}-${STAMP}.tgz" \
    --project "${PROJECT}" \
    --zone "${ZONE}" || true
fi

if [[ "${DESTROY_INSTANCE}" == "true" ]]; then
  echo "Deleting instance: ${INSTANCE}"
  gcloud compute instances delete "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" --quiet || true
elif [[ "${KEEP_INSTANCE}" == "true" ]]; then
  echo "Keeping instance running: ${INSTANCE}"
else
  echo "Stopping instance: ${INSTANCE}"
  gcloud compute instances stop "${INSTANCE}" --project "${PROJECT}" --zone "${ZONE}" >/dev/null || true
fi

echo "Done. Local artifacts directory: ${OUTPUT_DIR}"

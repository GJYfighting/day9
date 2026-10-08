#!/usr/bin/env bash
# Source this file before every Day9 command.

DAY9_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAY9_RUNTIME_ROOT="$DAY9_ROOT/runtime"
DAY9_PREVIOUS_NAME="day""8"
DAY9_PREVIOUS_ROOT="$(dirname "$DAY9_ROOT")/$DAY9_PREVIOUS_NAME"

if [[ "${CONDA_SHLVL:-0}" != "0" || -n "${CONDA_PREFIX:-}" ]]; then
  echo "检测到Anaconda环境。请先执行：conda deactivate" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if declare -F deactivate >/dev/null 2>&1; then
    deactivate
  else
    unset VIRTUAL_ENV
  fi
fi

DAY9_CLEAN_PATH=""
IFS=: read -ra DAY9_PATH_PARTS <<<"$PATH"
for DAY9_PATH_PART in "${DAY9_PATH_PARTS[@]}"; do
  [[ "$DAY9_PATH_PART" == *anaconda* || "$DAY9_PATH_PART" == *conda* ]] && continue
  [[ "$DAY9_PATH_PART" == "$DAY9_PREVIOUS_ROOT"* ]] && continue
  [[ "$DAY9_PATH_PART" =~ /day[0-8](/|$|-) ]] && continue
  DAY9_CLEAN_PATH="${DAY9_CLEAN_PATH:+$DAY9_CLEAN_PATH:}$DAY9_PATH_PART"
done
export PATH="/usr/bin:$DAY9_CLEAN_PATH"
unset DAY9_PATH_PART DAY9_PATH_PARTS DAY9_CLEAN_PATH
unset PYTHONHOME
unset PYTHONNOUSERSITE

source /opt/ros/humble/setup.bash
source "$DAY9_ROOT/../install/setup.bash"

DAY9_CLEAN_PYTHONPATH=""
IFS=: read -ra DAY9_PYTHONPATH_PARTS <<<"${PYTHONPATH:-}"
for DAY9_PYTHONPATH_PART in "${DAY9_PYTHONPATH_PARTS[@]}"; do
  [[ "$DAY9_PYTHONPATH_PART" == "$DAY9_PREVIOUS_ROOT"* ]] && continue
  [[ "$DAY9_PYTHONPATH_PART" =~ /day[0-8](/|$|-) ]] && continue
  DAY9_CLEAN_PYTHONPATH="${DAY9_CLEAN_PYTHONPATH:+$DAY9_CLEAN_PYTHONPATH:}$DAY9_PYTHONPATH_PART"
done
export PYTHONPATH="$DAY9_ROOT/vendor:$DAY9_ROOT${DAY9_CLEAN_PYTHONPATH:+:$DAY9_CLEAN_PYTHONPATH}"
unset DAY9_PYTHONPATH_PART DAY9_PYTHONPATH_PARTS DAY9_CLEAN_PYTHONPATH
unset DAY9_PREVIOUS_NAME DAY9_PREVIOUS_ROOT

mkdir -p \
  "$DAY9_ROOT/results" \
  "$DAY9_RUNTIME_ROOT/ros-home" \
  "$DAY9_RUNTIME_ROOT/logs/ros" \
  "$DAY9_RUNTIME_ROOT/logs/check" \
  "$DAY9_RUNTIME_ROOT/ignition/log" \
  "$DAY9_RUNTIME_ROOT/ignition/fuel" \
  "$DAY9_RUNTIME_ROOT/pycache" \
  "$DAY9_RUNTIME_ROOT/xdg/cache" \
  "$DAY9_RUNTIME_ROOT/xdg/config" \
  "$DAY9_RUNTIME_ROOT/xdg/data" \
  "$DAY9_RUNTIME_ROOT/matplotlib" \
  "$DAY9_RUNTIME_ROOT/qml-cache" \
  "$DAY9_RUNTIME_ROOT/torch" \
  "$DAY9_RUNTIME_ROOT/cuda-cache" \
  "$DAY9_RUNTIME_ROOT/tmp"

export DAY9_ROOT DAY9_RUNTIME_ROOT
export DAY9_PYTHON=/usr/bin/python3
export CAMERA_TYPE=GEMINI
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export ROS_DOMAIN_ID="${DAY9_ROS_DOMAIN_ID:-91}"
export IGN_PARTITION="${DAY9_IGN_PARTITION:-day9_${USER}}"
# All Day9 simulator and CLI processes are local; avoid VM NIC discovery.
export IGN_IP="${DAY9_IGN_IP:-127.0.0.1}"
export ROS_HOME="$DAY9_RUNTIME_ROOT/ros-home"
export ROS_LOG_DIR="$DAY9_RUNTIME_ROOT/logs/ros"
export IGN_LOG_PATH="$DAY9_RUNTIME_ROOT/ignition/log"
export IGN_FUEL_CACHE_PATH="$DAY9_RUNTIME_ROOT/ignition/fuel"
export PYTHONPYCACHEPREFIX="$DAY9_RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$DAY9_RUNTIME_ROOT/xdg/cache"
export XDG_CONFIG_HOME="$DAY9_RUNTIME_ROOT/xdg/config"
export XDG_DATA_HOME="$DAY9_RUNTIME_ROOT/xdg/data"
export MPLCONFIGDIR="$DAY9_RUNTIME_ROOT/matplotlib"
export QML_DISK_CACHE_PATH="$DAY9_RUNTIME_ROOT/qml-cache"
export TORCH_HOME="$DAY9_RUNTIME_ROOT/torch"
export CUDA_CACHE_PATH="$DAY9_RUNTIME_ROOT/cuda-cache"
export TMPDIR="$DAY9_RUNTIME_ROOT/tmp"

export XDG_RUNTIME_DIR="$DAY9_RUNTIME_ROOT/xdg/run"
mkdir -p "$XDG_RUNTIME_DIR" "$DAY9_RUNTIME_ROOT/home"
chmod 700 "$XDG_RUNTIME_DIR"
export DAY9_APPLICATION_HOME="$DAY9_RUNTIME_ROOT/home"
# A Ruby shim is required because ros_gz_sim invokes `ruby <ign executable>`.
# Give Ignition a real private application home without repurposing shell HOME.
mkdir -p "$DAY9_RUNTIME_ROOT/bin"
DAY9_IGN_SHIM_TMP="$(mktemp "$DAY9_RUNTIME_ROOT/bin/.ign.XXXXXX")"
cat > "$DAY9_IGN_SHIM_TMP" <<'DAY9_IGN_RUBY'
#!/usr/bin/ruby
ENV["HOME"] = ENV.fetch("DAY9_APPLICATION_HOME")
load "/usr/bin/ign"
DAY9_IGN_RUBY
chmod 755 "$DAY9_IGN_SHIM_TMP"
mv -f "$DAY9_IGN_SHIM_TMP" "$DAY9_RUNTIME_ROOT/bin/ign"
unset DAY9_IGN_SHIM_TMP
export PATH="$DAY9_RUNTIME_ROOT/bin:$PATH"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export IGN_HOMEDIR="$DAY9_RUNTIME_ROOT/home"
export GZ_HOMEDIR="$DAY9_RUNTIME_ROOT/home"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export ROS_LOCALHOST_ONLY=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$DAY9_ROOT/configs/dds_udp.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export TORCHINDUCTOR_CACHE_DIR="$DAY9_RUNTIME_ROOT/torchinductor"
export JOBLIB_TEMP_FOLDER="$TMPDIR"
# Pin the CPU thread counts observed during the Day9 baseline (not a network change).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$DAY9_RUNTIME_ROOT/numba"
export TRITON_CACHE_DIR="$DAY9_RUNTIME_ROOT/triton"
export __GL_SHADER_DISK_CACHE_PATH="$DAY9_RUNTIME_ROOT/gl-cache"
cd "$DAY9_ROOT"

echo "DAY9_ENV=READY"
echo "ROOT=$DAY9_ROOT"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "IGN_PARTITION=$IGN_PARTITION"
echo "PYTHON=$DAY9_PYTHON"

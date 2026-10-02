#!/usr/bin/env bash
set -euo pipefail

# Build the exact upstream GEDI simulator used in Phase 5 on macOS/conda.
# Rebuilding changes binary hashes; rerun the smoke test and all lidar stages afterwards.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXTERNAL="$ROOT/data/external"
GEDI="$EXTERNAL/gedisimulator"
LIBCLIDAR="$EXTERNAL/libclidar"
TOOLS="$EXTERNAL/hancock-tools"
GEDI_COMMIT=49ad4f26b03083b61ee0bd424e8a7e4a8f2c301d
LIBCLIDAR_COMMIT=8535ed4428f71445f06c01981e6b9758a53555b3
TOOLS_COMMIT=506895e2a3968bdb4f96bb193b31963b8a68a789

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the project conda environment first." >&2
  exit 1
fi

clone_or_verify() {
  local url=$1
  local directory=$2
  local commit=$3
  if [[ ! -d "$directory/.git" ]]; then
    git clone "$url" "$directory"
    git -C "$directory" checkout --detach "$commit"
  fi
  if [[ $(git -C "$directory" rev-parse HEAD) != "$commit" ]]; then
    echo "Unexpected commit in $directory; refusing to modify it." >&2
    exit 1
  fi
  if [[ -n $(git -C "$directory" status --porcelain) ]]; then
    echo "Source tree is dirty: $directory" >&2
    exit 1
  fi
}

mkdir -p "$EXTERNAL"
clone_or_verify https://bitbucket.org/StevenHancock/gedisimulator.git "$GEDI" "$GEDI_COMMIT"
clone_or_verify https://bitbucket.org/StevenHancock/libclidar.git "$LIBCLIDAR" "$LIBCLIDAR_COMMIT"
clone_or_verify https://bitbucket.org/StevenHancock/tools.git "$TOOLS" "$TOOLS_COMMIT"

CC=${CC:-clang}
ARCH=$(uname -m)
CFLAGS=(-O3 -Wall "-D$ARCH")
INCLUDES=(
  "-I$GEDI"
  "-I$GEDI/packages/cmpfit-1.2"
  "-I$LIBCLIDAR"
  "-I$TOOLS"
  "-I$CONDA_PREFIX/include"
  "-I$CONDA_PREFIX/include/gdal"
)
LIBS=(
  "-L$CONDA_PREFIX/lib"
  "-Wl,-rpath,$CONDA_PREFIX/lib"
  -lm -lgsl -lgslcblas -ltiff -lgeotiff -lhdf5 -lgdal
)

"$CC" "${CFLAGS[@]}" "${INCLUDES[@]}" -c \
  "$GEDI/packages/cmpfit-1.2/mpfit.c" -o "$GEDI/mpfit.o"

libclidar_sources=(
  libLasProcess libLasRead tiffWrite gaussFit libLidVoxel libTLSread
  libLidarHDF libOctree
)
for source in "${libclidar_sources[@]}"; do
  "$CC" "${CFLAGS[@]}" "${INCLUDES[@]}" -c \
    "$LIBCLIDAR/$source.c" -o "$LIBCLIDAR/$source.o"
done

gedi_common_sources=(gediIO gediNoise photonCount)
for source in "${gedi_common_sources[@]}"; do
  "$CC" "${CFLAGS[@]}" "${INCLUDES[@]}" -c \
    "$GEDI/$source.c" -o "$GEDI/$source.o"
done

objects=(
  "$GEDI/mpfit.o"
  "$LIBCLIDAR/libLasProcess.o"
  "$LIBCLIDAR/libLasRead.o"
  "$LIBCLIDAR/tiffWrite.o"
  "$LIBCLIDAR/gaussFit.o"
  "$LIBCLIDAR/libLidVoxel.o"
  "$LIBCLIDAR/libTLSread.o"
  "$LIBCLIDAR/libLidarHDF.o"
  "$LIBCLIDAR/libOctree.o"
  "$GEDI/gediIO.o"
  "$GEDI/gediNoise.o"
  "$GEDI/photonCount.o"
)

for program in gediRat gediMetric; do
  "$CC" "${CFLAGS[@]}" "${INCLUDES[@]}" -c \
    "$GEDI/$program.c" -o "$GEDI/$program.o"
  # Both program entry points include tools.c directly; linking tools.o duplicates symbols.
  "$CC" "${CFLAGS[@]}" "${objects[@]}" "$GEDI/$program.o" \
    -o "$GEDI/$program" "${LIBS[@]}"
  "$GEDI/$program" -help >/dev/null
done

echo "Built $GEDI/gediRat and $GEDI/gediMetric"
echo "Next: python scripts/smoke_test_gedi_simulator.py"

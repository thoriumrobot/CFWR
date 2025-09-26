#!/usr/bin/env bash
set -euo pipefail

# Simple placeholder Soot-based slicer interface
# Accepts:
#   --projectRoot <path>
#   --targetFile <relative-or-abs .java>
#   --line <number>
#   --output <dir>
#   --member <class#sig>
#   --decompiler <vineflower.jar> (optional)

PROJECT_ROOT=""
TARGET_FILE=""
LINE_NO=""
OUTPUT_DIR=""
MEMBER_SIG=""
DECOMPILER_JAR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --projectRoot) PROJECT_ROOT="$2"; shift 2;;
    --targetFile)  TARGET_FILE="$2"; shift 2;;
    --line)        LINE_NO="$2"; shift 2;;
    --output)      OUTPUT_DIR="$2"; shift 2;;
    --member)      MEMBER_SIG="$2"; shift 2;;
    --decompiler)  DECOMPILER_JAR="$2"; shift 2;;
    *) echo "[soot_slicer] Unknown arg: $1"; shift 1;;
  esac
done

if [[ -z "$PROJECT_ROOT" || -z "$TARGET_FILE" || -z "$LINE_NO" || -z "$OUTPUT_DIR" || -z "$MEMBER_SIG" ]]; then
  echo "[soot_slicer] Missing required args" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# Normalize TARGET_FILE to absolute
if [[ "$TARGET_FILE" != /* ]]; then
  SRC_ABS="${PROJECT_ROOT%/}/$TARGET_FILE"
else
  SRC_ABS="$TARGET_FILE"
fi

if [[ ! -f "$SRC_ABS" ]]; then
  echo "[soot_slicer] Source file not found: $SRC_ABS" >&2
  # Create a placeholder slice to avoid downstream blanks
  echo "// placeholder slice (source not found) for $MEMBER_SIG" > "$OUTPUT_DIR/slice.java"
  exit 0
fi

# Heuristic: copy the source file into the output dir as a minimal slice
BASENAME=$(basename "$SRC_ABS")
cp -f "$SRC_ABS" "$OUTPUT_DIR/$BASENAME"

# Also write a small metadata file to help debugging
cat > "$OUTPUT_DIR/slice.meta" << META
member=$MEMBER_SIG
line=$LINE_NO
source=$SRC_ABS
META

echo "[soot_slicer] Wrote slice: $OUTPUT_DIR/$BASENAME"
exit 0

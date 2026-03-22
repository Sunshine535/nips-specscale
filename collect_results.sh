#!/bin/bash
# =============================================================================
# Collect and package all experiment results for archival / git push / transfer
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_NAME="$(basename "$PROJ_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

ARCHIVE_DIR="$PROJ_DIR/results_archive"
mkdir -p "$ARCHIVE_DIR"

echo "============================================"
echo " Collecting results: $PROJ_NAME"
echo " Timestamp: $TIMESTAMP"
echo "============================================"

DIRS_TO_COLLECT=()
for d in results outputs checkpoints logs; do
    if [ -d "$PROJ_DIR/$d" ] && [ "$(find "$PROJ_DIR/$d" -type f 2>/dev/null | head -1)" ]; then
        DIRS_TO_COLLECT+=("$d")
    fi
done

if [ ${#DIRS_TO_COLLECT[@]} -eq 0 ]; then
    echo "No results found to collect."
    exit 0
fi

echo "Directories to archive: ${DIRS_TO_COLLECT[*]}"

ARCHIVE_NAME="${PROJ_NAME}_results_${TIMESTAMP}"
SNAPSHOT_DIR="$ARCHIVE_DIR/$ARCHIVE_NAME"
mkdir -p "$SNAPSHOT_DIR"

for d in "${DIRS_TO_COLLECT[@]}"; do
    echo "  Copying $d/ ..."
    cp -r "$PROJ_DIR/$d" "$SNAPSHOT_DIR/$d"
done

# Generate summary
python3 -c "
import json, os, glob

snapshot = '$SNAPSHOT_DIR'
summary = {
    'project': '$PROJ_NAME',
    'timestamp': '$TIMESTAMP',
    'directories': [],
}

for d in os.listdir(snapshot):
    dp = os.path.join(snapshot, d)
    if os.path.isdir(dp):
        files = []
        for root, dirs, fnames in os.walk(dp):
            for fn in fnames:
                fp = os.path.join(root, fn)
                files.append({
                    'path': os.path.relpath(fp, snapshot),
                    'size_kb': round(os.path.getsize(fp) / 1024, 1),
                })
        summary['directories'].append({
            'name': d,
            'file_count': len(files),
            'total_size_mb': round(sum(f['size_kb'] for f in files) / 1024, 1),
        })
        json_files = [f for f in files if f['path'].endswith('.json')]
        summary['directories'][-1]['json_files'] = len(json_files)

with open(os.path.join(snapshot, 'MANIFEST.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
" 2>/dev/null || echo "  (manifest generation skipped)"

echo ""
echo "Creating tarball..."
cd "$ARCHIVE_DIR"
tar czf "${ARCHIVE_NAME}.tar.gz" "$ARCHIVE_NAME"
rm -rf "$ARCHIVE_NAME"

ARCHIVE_PATH="$ARCHIVE_DIR/${ARCHIVE_NAME}.tar.gz"
SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
echo ""
echo "============================================"
echo " Archive created: $ARCHIVE_PATH"
echo " Size: $SIZE"
echo "============================================"
echo ""
echo "To transfer:"
echo "  scp $ARCHIVE_PATH user@host:/path/"
echo ""
echo "To push results to git:"
echo "  git add results/ && git commit -m 'Add experiment results $TIMESTAMP'"

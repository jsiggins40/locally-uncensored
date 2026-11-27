#!/usr/bin/env bash
# Checks why "Recent edits" on the edit page shows empty tiles: are the result files there,
# can the VM read them, and does it hand them (and their small previews) to the page?
# Changes nothing. Prints file names, sizes and status codes only, never a picture or a prompt.
#   rv6 recent-check
OUT="$HOME/ComfyUI/output"
echo "== Page on this VM: $(grep -o "PAGE_BUILD = '[^']*'" "$HOME/ComfyUI/custom_nodes/rv6_mobile/web/edit.html" 2>/dev/null | head -1 | cut -d"'" -f2)"
[ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' 127.0.0.1:8188/)" = 200 ] && echo "  OK    ComfyUI answers" || { echo "  PROBLEM  ComfyUI does not answer. Run: edit-restart"; exit 0; }
n=$(ls "$OUT" 2>/dev/null | grep -cE '^rv6_(edit|cloud)_')
echo "== Result files in ~/ComfyUI/output: $n (free disk: $(df --output=avail -BG "$OUT" 2>/dev/null | tail -1 | tr -dc 0-9) GB)"
[ "$n" -gt 0 ] || { echo "  There are none on this VM, so Recent edits has nothing to show. Results from another VM or an older disk are not here."; exit 0; }
list=$(curl -s -m 20 "127.0.0.1:8188/rv6m/recent?limit=3")
echo "$list" | python3 -c 'import json,sys
try: j = json.load(sys.stdin); print("  OK    the page gets a list of", j.get("total"), "results")
except Exception: print("  PROBLEM  the list for the page did not come back (old page code? run edit-restart)")'
echo "== The newest three: can the VM read them, and what does the page get?"
ls -t "$OUT" | grep -E '^rv6_(edit|cloud)_' | head -3 | while read -r f; do
  size=$(stat -c %s "$OUT/$f" 2>/dev/null || echo 0)
  kind=$(python3 - "$OUT/$f" <<'PY'
import sys
b = open(sys.argv[1], "rb").read(16)
print("PNG" if b[:4] == b"\x89PNG" else "JPEG" if b[:2] == b"\xff\xd8" else "WEBP" if b[8:12] == b"WEBP" else "MP4" if b[4:8] == b"ftyp" else "EMPTY" if not b else "not a picture or video")
PY
)
  q=$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$f")
  full=$(curl -s -o /dev/null -m 60 -w '%{http_code}, %{size_download} bytes' "127.0.0.1:8188/api/view?filename=$q&type=output")
  small=$(curl -s -o /dev/null -m 120 -w '%{http_code}, %{size_download} bytes, %{content_type}' "127.0.0.1:8188/rv6m/jpeg?filename=$q&subfolder=&max=600")
  echo "  $f"
  echo "      on disk: $size bytes, $kind | the file itself: $full | its small preview: $small"
done
echo "== What to look for: both codes should be 200 with more than 0 bytes. A 0-byte or EMPTY file was saved broken;"
echo "   a preview that is not 200 means the VM can't make it (the error is in ~/comfyui.log). Send this whole output."

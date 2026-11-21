#!/usr/bin/env bash
# Shows what is using the VM's disk, biggest first, and what could be deleted.
# Read-only: it deletes nothing.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/disk-report.sh)
C="$HOME/ComfyUI"; M="$C/models"
gb() { awk -v h="$HOME/" '{n = $0; sub(/^[0-9]+[ \t]+/, "", n); sub(h, "", n); printf "%8.1f GB  %s\n", $1 / 1048576, n}'; }
size() { du -sk "$@" 2>/dev/null | sort -nr | gb; }
one() { [ -e "$1" ] && printf "%8.1f GB  %s\n" "$(du -sk "$1" 2>/dev/null | awk '{print $1 / 1048576}')" "$2"; }

echo "== Disk"
df -h --output=size,used,avail,pcent "$HOME" | sed 's/^/  /'

echo; echo "== Image models, by kind"
size "$M"/* | head -15

echo; echo "== Everything else"
{ du -sk "$HOME/llm" "$HOME/open-webui-data" "$HOME/llama.cpp" "$HOME/searxng" "$HOME/.local" "$C/venv" "$C/output" "$C/input" "$C/temp" "$C/custom_nodes" 2>/dev/null
  du -sk "$HOME"/.cache/* 2>/dev/null; } | sort -nr | gb | head -14

echo; echo "== The 30 biggest model files"
find "$M" "$HOME/llm" -type f -size +400M -printf '%k %P\n' 2>/dev/null | sort -nr | head -30 | gb

echo; echo "== Could be deleted"
echo "  Leftovers and caches (safe: nothing uses them, or they are fetched again when needed)"
one "$M/_hf_tmp" "unfinished downloads (models/_hf_tmp)"
one "$HOME/.cache/pip" "installer cache (~/.cache/pip)"
one "$HOME/.cache/uv" "installer cache (~/.cache/uv)"
one "$C/temp" "ComfyUI preview pictures (ComfyUI/temp)"
t=$(find "$C/output" -maxdepth 1 -type f -name 'rv6m_*test*' 2>/dev/null | wc -l)
[ "$t" -gt 0 ] && echo "            $t installer test pictures in ComfyUI/output (rv6m_*test*)"
echo "  Models that failed their test and were set aside"
one "$M/_disabled" "models/_disabled: $(ls "$M/_disabled" 2>/dev/null | tr '\n' ' ')"
echo "  Yours: only delete what you have saved elsewhere or don't want"
n=$(find "$C/output" -type f 2>/dev/null | wc -l); one "$C/output" "$n finished pictures and videos (ComfyUI/output)"
n=$(find "$C/input" -type f 2>/dev/null | wc -l); one "$C/input" "$n photos uploaded for editing (ComfyUI/input)"
one "$HOME/open-webui-data" "your chats and the chat page's own models (open-webui-data)"
echo
echo "Nothing was deleted. Send this list back to decide what goes."

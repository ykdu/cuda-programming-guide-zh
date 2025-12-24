#!/usr/bin/env bash
set -euo pipefail

# macOS system bash is usually 3.2 -> NO associative arrays.
# We use parallel arrays to keep ordering deterministic.

CHAPTER_DIRS=(
  "01-introduction-to-cuda"
  "02-programming-gpus-in-cuda"
  "03-advanced-cuda"
  "04-cuda-features"
  "05-technical-appendices"
  "06-notices"
)

CHAPTER_TITLES=(
  "第 1 章 Introduction to CUDA"
  "第 2 章 Programming GPUs in CUDA"
  "第 3 章 Advanced CUDA"
  "第 4 章 CUDA Features"
  "第 5 章 Technical Appendices"
  "第 6 章 Notices"
)

# Generate chapter README listing all md files except README.md
gen_chapter_readme() {
  local dir="$1"
  local title="$2"

  # Collect md files (sorted), excluding README.md
  local files
  files=$(ls -1 "${dir}"/*.md 2>/dev/null | sed 's|.*/||' | grep -v '^README\.md$' | sort || true)

  {
    echo "# ${title}"
    echo
    echo "> 目录由脚本自动生成。正文内容请进入各小节文件。"
    echo
    if [[ -z "${files}" ]]; then
      echo "_（暂无小节文件）_"
    else
      while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        # Use first line heading inside file if available, else fallback to filename
        local heading
        heading=$(head -n 1 "${dir}/${f}" 2>/dev/null | sed -E 's/^#\s+//')
        [[ -z "${heading}" ]] && heading="${f}"
        echo "- [${heading}](${f})"
      done <<< "${files}"
    fi
  } > "${dir}/README.md"
}

# Generate root BOOK.md
gen_book() {
  local out="BOOK.md"
  {
    echo "# CUDA Programming Guide 中文版（总目录）"
    echo
    echo "> 本目录由脚本自动生成，用于 GitHub 上快速导航。"
    echo
  } > "${out}"

  local i
  for i in "${!CHAPTER_DIRS[@]}"; do
    local dir="${CHAPTER_DIRS[$i]}"
    local title="${CHAPTER_TITLES[$i]}"

    echo "## [${title}](${dir}/README.md)" >> "${out}"
    echo >> "${out}"

    local files
    files=$(ls -1 "${dir}"/*.md 2>/dev/null | sed 's|.*/||' | grep -v '^README\.md$' | sort || true)

    if [[ -z "${files}" ]]; then
      echo "_（暂无小节文件）_" >> "${out}"
      echo >> "${out}"
      continue
    fi

    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      local heading
      heading=$(head -n 1 "${dir}/${f}" 2>/dev/null | sed -E 's/^#\s+//')
      [[ -z "${heading}" ]] && heading="${f}"
      echo "- [${heading}](${dir}/${f})" >> "${out}"
    done <<< "${files}"

    echo >> "${out}"
  done
}

main() {
  local i
  for i in "${!CHAPTER_DIRS[@]}"; do
    local dir="${CHAPTER_DIRS[$i]}"
    local title="${CHAPTER_TITLES[$i]}"

    if [[ -d "${dir}" ]]; then
      gen_chapter_readme "${dir}" "${title}"
      echo "Generated: ${dir}/README.md"
    else
      echo "Skip (missing dir): ${dir}"
    fi
  done

  gen_book
  echo "Generated: BOOK.md"
}

main "$@"

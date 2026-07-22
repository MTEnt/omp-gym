#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: package-release.sh --version <semver> --target <rust-target> --binary <path> --output <directory>
EOF
  exit 2
}

version=''
target=''
binary=''
output=''
while (($# > 0)); do
  case "$1" in
    --version)
      (($# >= 2)) || usage
      version=$2
      shift 2
      ;;
    --target)
      (($# >= 2)) || usage
      target=$2
      shift 2
      ;;
    --binary)
      (($# >= 2)) || usage
      binary=$2
      shift 2
      ;;
    --output)
      (($# >= 2)) || usage
      output=$2
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]] || {
  printf 'invalid release version: %s\n' "$version" >&2
  exit 2
}
[[ "$target" =~ ^[0-9A-Za-z_.-]+$ ]] || {
  printf 'invalid Rust target: %s\n' "$target" >&2
  exit 2
}
[[ -x "$binary" ]] || {
  printf 'release binary is missing or not executable: %s\n' "$binary" >&2
  exit 1
}
[[ -n "$output" ]] || usage

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for required_file in LICENSE README.md extensions/omp/gym.ts; do
  [[ -f "$repo_root/$required_file" ]] || {
    printf 'required package file is missing: %s\n' "$required_file" >&2
    exit 1
  }
done

actual_version=$("$binary" --version)
expected_version="omp-gym $version"
[[ "$actual_version" == "$expected_version" ]] || {
  printf 'binary version mismatch: expected %q, got %q\n' "$expected_version" "$actual_version" >&2
  exit 1
}

package_name="omp-gym-v${version}-${target}"
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/omp-gym-package.XXXXXX")
trap 'rm -rf "$staging_root"' EXIT
package_root="$staging_root/$package_name"
mkdir -p "$package_root/extensions/omp" "$output"
install -m 755 "$binary" "$package_root/omp-gym"
install -m 644 "$repo_root/LICENSE" "$package_root/LICENSE"
install -m 644 "$repo_root/README.md" "$package_root/README.md"
install -m 644 "$repo_root/extensions/omp/gym.ts" "$package_root/extensions/omp/gym.ts"

tar -czf "$output/$package_name.tar.gz" -C "$staging_root" "$package_name"
printf '%s\n' "$output/$package_name.tar.gz"

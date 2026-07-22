#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

fake_binary="$temp_dir/omp-gym"
cat >"$fake_binary" <<'EOF'
#!/usr/bin/env bash
printf 'omp-gym 0.1.0-test\n'
EOF
chmod +x "$fake_binary"

output_dir="$temp_dir/dist"
"$repo_root/scripts/package-release.sh" \
  --version 0.1.0-test \
  --target aarch64-apple-darwin \
  --binary "$fake_binary" \
  --output "$output_dir"

archive="$output_dir/omp-gym-v0.1.0-test-aarch64-apple-darwin.tar.gz"
test -f "$archive"

package_dir="omp-gym-v0.1.0-test-aarch64-apple-darwin"
expected_entries=$(printf '%s\n' \
  "$package_dir/" \
  "$package_dir/LICENSE" \
  "$package_dir/README.md" \
  "$package_dir/extensions/" \
  "$package_dir/extensions/omp/" \
  "$package_dir/extensions/omp/gym.ts" \
  "$package_dir/omp-gym")
actual_entries=$(tar -tzf "$archive" | LC_ALL=C sort)
expected_entries=$(printf '%s\n' "$expected_entries" | LC_ALL=C sort)
if [[ "$actual_entries" != "$expected_entries" ]]; then
  printf 'unexpected archive contents\nexpected:\n%s\nactual:\n%s\n' "$expected_entries" "$actual_entries" >&2
  exit 1
fi

extract_dir="$temp_dir/extract"
mkdir -p "$extract_dir"
tar -xzf "$archive" -C "$extract_dir"
test -x "$extract_dir/$package_dir/omp-gym"
test "$("$extract_dir/$package_dir/omp-gym")" = 'omp-gym 0.1.0-test'
cmp "$repo_root/LICENSE" "$extract_dir/$package_dir/LICENSE"
cmp "$repo_root/README.md" "$extract_dir/$package_dir/README.md"
cmp "$repo_root/extensions/omp/gym.ts" "$extract_dir/$package_dir/extensions/omp/gym.ts"

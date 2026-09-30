#!/usr/bin/env bash

# Prepare a local release branch for the single LexNLP repository.
#
# This deliberately does not fetch, pull, commit, tag, or push.  Review and
# commit the resulting branch before publishing it through the normal release
# process.

set -euo pipefail

readonly SCRIPT_NAME="${0##*/}"

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME VERSION [BRANCH]

Create a local release branch from the current clean checkout, set VERSION in
pyproject.toml and lexnlp/__init__.py, and refresh uv.lock without upgrading
dependencies. BRANCH defaults to release/VERSION.

The command never fetches, pulls, commits, tags, or pushes.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi

[[ $# -ge 1 && $# -le 2 ]] || {
    usage >&2
    exit 2
}

release_version=$1
release_branch=${2:-"release/$release_version"}

# Accept normal PEP 440 release, pre-release, post-release, development, and
# local-version forms. Requiring an explicit version prevents accidental
# branch names such as "latest" from becoming package versions.
if ! [[ $release_version =~ ^([0-9]+)(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?(\+[A-Za-z0-9]+([._-][A-Za-z0-9]+)*)?$ ]]; then
    die "VERSION must be a PEP 440-style version (for example, 2.3.1 or 2.4.0rc1)"
fi

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "must be run from a Git checkout"
cd "$repo_root"

[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || die "working tree is not clean"
git check-ref-format --branch "$release_branch" >/dev/null || die "invalid branch name: $release_branch"
git show-ref --verify --quiet "refs/heads/$release_branch" && die "local branch already exists: $release_branch"
[[ -z $(git branch --remotes --list "*/$release_branch") ]] || die "remote-tracking branch already exists: $release_branch"

command -v python3 >/dev/null || die "python3 is required"
command -v uv >/dev/null || die "uv is required to refresh uv.lock"

readonly pyproject_path="$repo_root/pyproject.toml"
readonly package_init_path="$repo_root/lexnlp/__init__.py"
readonly lock_path="$repo_root/uv.lock"
[[ -f $pyproject_path ]] || die "missing pyproject.toml"
[[ -f $package_init_path ]] || die "missing lexnlp/__init__.py"
[[ -f $lock_path ]] || die "missing uv.lock"

read_version_source() {
    local source=$1
    python3 - "$pyproject_path" "$package_init_path" "$source" <<'PY'
import re
import sys
from pathlib import Path

pyproject_path, init_path, source = map(Path, sys.argv[1:])
if source.name == "pyproject":
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    project_section = re.search(r"(?ms)^\[project\][ \t]*\n(.*?)(?=^\[|\Z)", pyproject_text)
    if not project_section:
        raise SystemExit(f"missing [project] section in {pyproject_path}")
    matches = re.findall(r'^version[ \t]*=[ \t]*"([^"\\]+)"[ \t]*$', project_section.group(1), flags=re.MULTILINE)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one [project].version in {pyproject_path}")
    print(matches[0])
    raise SystemExit

init_text = init_path.read_text(encoding="utf-8")
pattern = {
    "runtime": r'^__version__\s*=\s*"([^"\\]+)"\s*$',
    "license": r'^__license__\s*=\s*"https://github\.com/LexPredict/lexpredict-lexnlp/blob/([^/"\\]+)/LICENSE"\s*$',
}[str(source)]
matches = re.findall(pattern, init_text, flags=re.MULTILINE)
if len(matches) != 1:
    raise SystemExit(f"expected exactly one {source} version in {init_path}")
print(matches[0])
PY
}

package_version=$(read_version_source pyproject) || die "could not read [project].version"
runtime_version=$(read_version_source runtime) || die "could not read lexnlp.__version__"
license_version=$(read_version_source license) || die "could not read the versioned license URL"

[[ $package_version == "$runtime_version" ]] || die "pyproject.toml ($package_version) and lexnlp.__version__ ($runtime_version) disagree"
[[ $package_version == "$license_version" ]] || die "lexnlp.__license__ ($license_version) does not match the package version ($package_version)"
[[ $release_version != "$package_version" ]] || die "VERSION is already $package_version"

git switch -c "$release_branch"

python3 - "$pyproject_path" "$package_init_path" "$package_version" "$release_version" <<'PY'
import re
import sys
from pathlib import Path

pyproject_path = Path(sys.argv[1])
init_path = Path(sys.argv[2])
old_version = sys.argv[3]
new_version = sys.argv[4]

pyproject_text = pyproject_path.read_text(encoding="utf-8")
old_project_version = f'version = "{old_version}"'
if pyproject_text.count(old_project_version) != 1:
    raise SystemExit(f"expected exactly one {old_project_version!r} in {pyproject_path}")
pyproject_path.write_text(
    pyproject_text.replace(old_project_version, f'version = "{new_version}"'), encoding="utf-8"
)

init_text = init_path.read_text(encoding="utf-8")
updates = {
    rf'(?m)^__version__\s*=\s*"{re.escape(old_version)}"\s*$': f'__version__ = "{new_version}"',
    rf'(?m)^__license__\s*=\s*"https://github\.com/LexPredict/lexpredict-lexnlp/blob/{re.escape(old_version)}/LICENSE"\s*$': (
        f'__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/{new_version}/LICENSE"'
    ),
}
for pattern, replacement in updates.items():
    init_text, substitutions = re.subn(pattern, replacement, init_text)
    if substitutions != 1:
        raise SystemExit(f"expected exactly one matching version source in {init_path}")
init_path.write_text(init_text, encoding="utf-8")
PY

# With a checked-in lockfile, `uv lock` retains the existing resolution unless
# the project metadata requires a change. Do not use `--upgrade` here.
uv lock
uv lock --locked

printf 'Prepared local branch %s for LexNLP %s. Review, test, commit, tag, and push it manually.\n' \
    "$release_branch" "$release_version"

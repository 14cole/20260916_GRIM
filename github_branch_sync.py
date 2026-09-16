"""Recreate a GitHub branch on this machine without git or a zip download.

Paste this file into an editor, save it as github_branch_sync.py, and run:

    python github_branch_sync.py 14cole/20260916_GRIM main C:\\path\\to\\folder

Only the Python standard library is used. The script reads the branch's file
list from the GitHub API, fetches each file's text from GitHub, and writes it
into the destination folder. Every file is checked against GitHub's hash, and
files that already match are skipped, so re-running it updates a folder in
place. Files marked ``eol=crlf`` in .gitattributes get Windows line endings,
as a git checkout would. Local files that are not on the branch are left alone.

Private repositories need a token: set GITHUB_TOKEN or pass --token.
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"


def request(url, token, *, accept=None, attempts=5):
    headers = {"User-Agent": "github-branch-sync"}
    if accept:
        headers["Accept"] = accept
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (429, 500, 502, 503, 504)
            if exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0":
                raise SystemExit(
                    "GitHub API rate limit reached; wait an hour or use a token."
                ) from exc
            if exc.code in (404, 422):
                raise SystemExit(
                    f"Not found: {url}\nCheck the repository and branch names "
                    "(private repositories need a token)."
                ) from exc
            if not retryable or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(2 ** attempt)


def api_json(path, token):
    return json.loads(request(API + path, token, accept="application/vnd.github+json"))


def git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def list_files(repo, branch, token):
    quoted_branch = urllib.parse.quote(branch, safe="")
    commit = api_json(f"/repos/{repo}/commits/{quoted_branch}", token)
    commit_sha = commit["sha"]
    tree = api_json(
        f"/repos/{repo}/git/trees/{commit['commit']['tree']['sha']}?recursive=1",
        token,
    )
    if tree.get("truncated"):
        raise SystemExit(
            "The branch has too many files for one GitHub tree listing; "
            "pass --prefix to recreate one folder at a time."
        )
    files, skipped = [], []
    for entry in tree["tree"]:
        if entry["type"] == "blob":
            files.append(entry)
        elif entry["type"] == "commit":
            skipped.append(f"{entry['path']} (submodule)")
    return commit_sha, files, skipped


def fetch_blob(repo, commit_sha, entry, token):
    if entry["mode"] == "120000":
        # Symbolic links are stored as their target path; write that text.
        data = base64.b64decode(
            api_json(f"/repos/{repo}/git/blobs/{entry['sha']}", token)["content"]
        )
    else:
        url = f"{RAW}/{repo}/{commit_sha}/{urllib.parse.quote(entry['path'])}"
        data = request(url, token)
    if git_blob_sha(data) != entry["sha"]:
        raise ValueError(f"{entry['path']}: content does not match GitHub's hash")
    return data


def load_eol_rules(repo, commit_sha, files, token):
    """Read ``eol``/``text`` settings from every .gitattributes on the branch.

    Git stores text with LF and converts it on checkout when a file is marked
    ``eol=crlf`` (Windows batch files, for example). Only that conversion is
    reproduced; macros other than ``binary`` are not expanded.
    """

    rules = []
    for entry in files:
        if entry["path"].rsplit("/", 1)[-1] != ".gitattributes":
            continue
        base = entry["path"][: -len(".gitattributes")]
        text = fetch_blob(repo, commit_sha, entry, token).decode("utf-8", "replace")
        for line in text.splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            value = None
            for attribute in parts[1:]:
                if attribute in ("-text", "binary"):
                    value = "none"
                elif attribute.startswith("eol="):
                    value = attribute[4:]
            if value is not None:
                rules.append((base, parts[0], value))
    # Deeper .gitattributes files take precedence over shallower ones.
    rules.sort(key=lambda rule: rule[0].count("/"))
    return rules


def eol_for(path, rules):
    result = None
    for base, pattern, value in rules:
        if not path.startswith(base):
            continue
        relative = path[len(base):]
        if "/" in pattern.strip("/"):
            matched = fnmatch.fnmatchcase(relative, pattern.lstrip("/"))
        else:
            matched = fnmatch.fnmatchcase(relative.rsplit("/", 1)[-1], pattern)
        if matched:
            result = value
    return result


def sync_file(repo, commit_sha, entry, destination, token, rules):
    path = entry["path"]
    target = os.path.join(destination, *path.split("/"))
    crlf = eol_for(path, rules) == "crlf"
    if os.path.isfile(target):
        with open(target, "rb") as stream:
            existing = stream.read()
        if crlf:
            existing = existing.replace(b"\r\n", b"\n")
        if git_blob_sha(existing) == entry["sha"]:
            return "unchanged", path

    data = fetch_blob(repo, commit_sha, entry, token)
    if crlf:
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    temporary = target + ".partial"
    with open(temporary, "wb") as stream:
        stream.write(data)
    os.replace(temporary, target)
    if entry["mode"] == "100755" and os.name != "nt":
        os.chmod(target, 0o755)
    return "written", path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="owner/name, e.g. 14cole/20260916_GRIM")
    parser.add_argument("branch", help="branch, tag, or commit, e.g. main")
    parser.add_argument("destination", help="folder to create or update")
    parser.add_argument("--prefix", default="",
                        help="only recreate files under this folder, e.g. GRIM_Backend/")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub token for private repositories")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    commit_sha, all_files, skipped = list_files(args.repo, args.branch, args.token)
    files = all_files
    prefix = args.prefix.strip("/")
    if prefix:
        files = [f for f in files if f["path"] == prefix or f["path"].startswith(prefix + "/")]
    if not files:
        raise SystemExit("No files matched.")
    destination = os.path.abspath(args.destination)
    rules = load_eol_rules(args.repo, commit_sha, all_files, args.token)
    print(f"{args.repo}@{args.branch} ({commit_sha[:7]}): {len(files)} files -> {destination}")

    counts = {"written": 0, "unchanged": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                sync_file, args.repo, commit_sha, entry, destination, args.token, rules
            ): entry
            for entry in files
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                status, _path = future.result()
                counts[status] += 1
            except Exception as exc:  # report every failure, keep going
                failures.append(f"{futures[future]['path']}: {exc}")
            print(f"\r{done}/{len(files)}", end="", flush=True)
    print()

    print(f"Written: {counts['written']}  Already up to date: {counts['unchanged']}")
    for item in skipped:
        print(f"Skipped {item}")
    if failures:
        print(f"{len(failures)} file(s) failed:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

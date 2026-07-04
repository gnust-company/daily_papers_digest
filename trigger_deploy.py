"""Trigger the GitHub Pages deploy workflow after a digest run.

Sends a `workflow_dispatch` event to start .github/workflows/deploy-website.yml.
That workflow pulls the latest digests from MinIO, builds the Docusaurus site,
and deploys to GitHub Pages.

This is the bridge between the two crons:
    your machine (digest) ---trigger--> GitHub (build + deploy web)

We use `workflow_dispatch` (not `repository_dispatch`) deliberately:
`actions/deploy-pages` has a known bug (actions/deploy-pages#383) where repeated
`repository_dispatch` runs against the same commit SHA silently keep the old
artifact — the deploy reports success but the live site never updates.
`workflow_dispatch` does not have this issue.

Environment variables (add to .env):
    GITHUB_TOKEN  A classic PAT with `repo` scope, or a fine-grained PAT with
                  "Actions: write" + "Contents: read" + "Metadata: read".
    GITHUB_REPO   owner/repo, e.g. gnust-company/daily_papers_digest
    GITHUB_REF    Branch the workflow runs on (default: main)

Usage:
    python trigger_deploy.py
Returns exit code 0 on success (HTTP 204), non-zero otherwise. Safe to call
at the end of the pipeline; failures here do not affect the digest itself.
"""
import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(override=False)


def trigger_deploy() -> bool:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO", "gnust-company/daily_papers_digest")
    ref = os.getenv("GITHUB_REF", "main")

    if not token:
        print("[trigger_deploy] GITHUB_TOKEN not set; skipping deploy trigger.")
        return False

    # workflow_dispatch endpoint: triggers .github/workflows/deploy-website.yml
    # on the configured branch. We use workflow_dispatch instead of
    # repository_dispatch because of actions/deploy-pages#383: repeated
    # repository_dispatch runs against the same commit SHA silently keep the
    # old Pages artifact, so the site never updates even though deploy reports
    # success.
    url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/"
        f"deploy-website.yml/dispatches"
    )
    payload = json.dumps({"ref": ref}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            # GitHub returns 204 No Content on success.
            if resp.status == 204:
                print(f"[trigger_deploy] OK — triggered deploy for {repo}.")
                return True
            print(f"[trigger_deploy] Unexpected status {resp.status}.")
            return False
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"[trigger_deploy] HTTP {e.code}: {body}")
        return False
    except Exception as e:  # noqa: BLE001 — never let this break the pipeline
        print(f"[trigger_deploy] Failed: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if trigger_deploy() else 1)

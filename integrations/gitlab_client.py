"""
integrations/gitlab_client.py

Handles:
- cloning/pulling the target repo locally (for RAG indexing + fixes)
- creating a fix branch, committing a proposed patch
- opening a Merge Request for human review (agent NEVER merges on its own)

Works with any language repo -- this layer is purely git/GitLab plumbing,
language-specific logic lives in the fix agent.
"""

from __future__ import annotations
import os
from pathlib import Path
import gitlab
import git  # GitPython


class GitLabClient:
    def __init__(self, url: str, private_token: str, clone_dir: str, default_branch: str = "main"):
        self.gl = gitlab.Gitlab(url, private_token=private_token)
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.default_branch = default_branch
        self.url = url
        self.private_token = private_token

    def ensure_local_clone(self, project_path: str) -> Path:
        """
        project_path e.g. 'group/subgroup/your-service-repo'
        Returns local path to the repo, cloning it if not already present,
        pulling latest if it is.
        """
        local_path = self.clone_dir / project_path.replace("/", "__")

        if local_path.exists():
            repo = git.Repo(local_path)
            repo.remotes.origin.fetch()
            repo.git.checkout(self.default_branch)
            repo.remotes.origin.pull()
        else:
            project = self.gl.projects.get(project_path)
            clone_url = project.http_url_to_repo.replace(
                "https://", f"https://oauth2:{self.private_token}@"
            )
            git.Repo.clone_from(clone_url, local_path, branch=self.default_branch)

        return local_path

    def create_fix_branch_and_commit(
        self,
        project_path: str,
        local_repo_path: Path,
        branch_name: str,
        file_changes: dict[str, str],
        commit_message: str,
    ) -> str:
        """
        file_changes: {relative_file_path: new_full_content}
        Returns the branch name that was pushed.
        """
        repo = git.Repo(local_repo_path)
        repo.git.checkout(self.default_branch)
        repo.git.pull()

        # create/reset branch
        if branch_name in repo.heads:
            repo.git.branch("-D", branch_name)
        repo.git.checkout("-b", branch_name)

        for rel_path, new_content in file_changes.items():
            full_path = local_repo_path / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(new_content, encoding="utf-8")
            repo.index.add([str(rel_path)])

        repo.index.commit(commit_message)
        origin = repo.remote(name="origin")
        origin.push(refspec=f"{branch_name}:{branch_name}", force=True)

        return branch_name

    def open_merge_request(
        self,
        project_path: str,
        source_branch: str,
        title: str,
        description: str,
        target_branch: str | None = None,
    ) -> str:
        """Opens a draft MR for human review. Returns the MR web URL."""
        project = self.gl.projects.get(project_path)
        mr = project.mergerequests.create(
            {
                "source_branch": source_branch,
                "target_branch": target_branch or self.default_branch,
                "title": f"Draft: {title}",
                "description": description,
                "remove_source_branch": True,
                "draft": True,
            }
        )
        return mr.web_url

    def get_recent_commits(self, local_repo_path: Path, file_path: str, limit: int = 10) -> list[dict]:
        """Used by the RCA agent to correlate recent changes with a bug."""
        repo = git.Repo(local_repo_path)
        commits = list(repo.iter_commits(paths=file_path, max_count=limit))
        return [
            {
                "hexsha": c.hexsha[:8],
                "author": c.author.name,
                "date": c.committed_datetime.isoformat(),
                "message": c.message.strip(),
            }
            for c in commits
        ]

from pathlib import Path
import pygit2 as git

from utils import _
import cli

cli.use_default_root()

clip = cli.patterns.experiment_tracking
clip.intro()
clip.libgit_version(
  major=git.LIBGIT2_VER_MAJOR,
  minor=git.LIBGIT2_VER_MINOR,
  revision=git.LIBGIT2_VER_REVISION)

repo_path = git.discover_repository(Path(__file__).parent)
repo = git.Repository(repo_path)
clip.repo_info(
  path=repo.workdir,
  refs=list(repo.references),
  branches=list(repo.branches),
  remotes=list(repo.remotes))

# Some of the code here has been taken, and/or adapted from the Hugging Face Transformers python package,
# which is under the Apache License, Version 2.0 (the "License").
"""
Rebuilding transformers.utils.hub ()
# TODO chose: either reimplement the cache + push_to_hub logic, or use/import transformers.utils.hub
Pros: no dependency, doesn't rely on transformers, a lot faster, will
Cons: more code here, might need to update it if changes has been made on the cache in the HF libraries
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from urllib.parse import urlparse
from uuid import uuid4

import requests
from requests.exceptions import HTTPError
from huggingface_hub import (
    CommitOperationAdd,
    create_branch,
    create_commit,
    create_repo,
    hf_hub_download,
    hf_hub_url,
)
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    build_hf_headers,
    hf_raise_for_status,
)
import tokenizers

from .constants import CURRENT_VERSION_PACKAGE


# taken from transformers.utils.import_utils
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
_is_offline_mode = (
    True
    if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES
    else False
)


def is_offline_mode():
    return _is_offline_mode


# Sets the cache directory
# By default, we use the same directory than the one of the Hugging Face Hub
# If a user downloads a model from the hub, all the files including the tokenizer will be stored in this directory
# In order to avoid pointlessly duplicated files, we will use the same directory
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    )
)
default_cache_path = os.path.join(hf_cache_home, "hub")
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)
MIDITOK_CACHE = os.getenv(
    "MIDITOK_CACHE", HUGGINGFACE_HUB_CACHE
)
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

_staging_mode = (
    os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
)
_default_endpoint = (
    "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"
)

HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get(
    "HF_ENDPOINT", _default_endpoint
)
HUGGINGFACE_CO_PREFIX = (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
)
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"
)

# Return value when trying to load a file from cache but the file does not exist in the distant repo.
_CACHED_NO_EXIST = object()


# TODO remove unused methods and constants


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape `(model_url,
    etag, size_MB)`. Filenames in `cache_dir` are use to get the metadata for each model, only urls ending with *.bin*
    are added.

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape `(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = MIDITOK_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []

    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = (f"miditok/{CURRENT_VERSION_PACKAGE}; tokenizers/{tokenizers.__version__}; "
          f"python/{sys.version.split()[0]}; session_id/{SESSION_ID}")
    if DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]):
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Optional[str]:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo on huggingface.co.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to `"main"` if it's not provided and no `commit_hash` is
            provided either.
        repo_type (`str`, *optional*):
            The type of the repo.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.
    """
    if revision is None:
        revision = "main"

    if cache_dir is None:
        cache_dir = MIDITOK_CACHE

    object_id = repo_id.replace("/", "--")
    if repo_type is None:
        repo_type = "model"
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None
    for subfolder in ["refs", "snapshots"]:
        if not os.path.isdir(os.path.join(repo_cache, subfolder)):
            return None

    # Resolve refs (for instance to convert main to the associated commit sha)
    cached_refs = os.listdir(os.path.join(repo_cache, "refs"))
    if revision in cached_refs:
        with open(os.path.join(repo_cache, "refs", revision)) as f:
            revision = f.read()

    if os.path.isfile(os.path.join(repo_cache, ".no_exist", revision, filename)):
        return _CACHED_NO_EXIST

    cached_shas = os.listdir(os.path.join(repo_cache, "snapshots"))
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    cached_file = os.path.join(repo_cache, "snapshots", revision, filename)
    return cached_file if os.path.isfile(cached_file) else None


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```"""

    # Private arguments
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
    if is_offline_mode() and not local_files_only:
        print("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
                )
            else:
                return None
        return resolved_file

    if cache_dir is None:
        cache_dir = MIDITOK_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(
            path_or_repo_id,
            full_filename,
            cache_dir=cache_dir,
            revision=_commit_hash,
            repo_type=repo_type,
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                raise EnvironmentError(
                    f"Could not locate {full_filename} inside {path_or_repo_id}."
                )

    user_agent = http_user_agent(user_agent)
    try:
        # Load from URL or cache if already cached
        resolved_file = hf_hub_download(
            path_or_repo_id,
            filename,
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to request access at "
            f"https://huggingface.co/{path_or_repo_id} and pass a token having permission to this repo either "
            "by logging in with `huggingface-cli login` or by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
            "having permission to this repo either by logging in with `huggingface-cli login` or by passing "
            "`token=<your_token>`"
        ) from e
    except RevisionNotFoundError as e:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
        ) from e
    except LocalEntryNotFoundError as e:
        # We try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision
        )
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        if (
            not _raise_exceptions_for_missing_entries
            or not _raise_exceptions_for_connection_errors
        ):
            return None
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
            " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ) from e
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        if revision is None:
            revision = "main"
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
        ) from e
    except HTTPError as err:
        # First we try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision
        )
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        if not _raise_exceptions_for_connection_errors:
            return None

        raise EnvironmentError(
            f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}"
        )

    return resolved_file


def get_file_from_repo(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("xlm-roberta-base", "tokenizer_config.json")
    ```"""

    return cached_file(
        path_or_repo_id=path_or_repo,
        filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )


def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """

    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    url = hf_hub_url(path_or_repo, filename=filename, revision=revision)
    headers = build_hf_headers(token=token, user_agent=http_user_agent())

    r = requests.head(
        url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10
    )
    try:
        hf_raise_for_status(r)
        return True
    except GatedRepoError as e:
        raise EnvironmentError(
            f"{path_or_repo} is a gated repository. Make sure to request access at "
            f"https://huggingface.co/{path_or_repo} and pass a token having permission to this repo either by "
            "logging in with `huggingface-cli login` or by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        raise EnvironmentError(
            f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'."
        )
    except RevisionNotFoundError as e:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        )
    except requests.HTTPError:
        # We return false for EntryNotFoundError (logical) as well as any connection error.
        return False


@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield working_dir


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a tokenizer to the Hugging Face hub.
    """

    @staticmethod
    def _create_repo(
        repo_id: str,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        Create the repo if needed, retrieves the token.
        """
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id

    @staticmethod
    def _upload_modified_files(
        file_path: Union[str, Path],
        repo_id: str,
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
    ):
        """
        Creates the commit to push.
        Instead of pushing the whole directory we only push the tokenizer config file just saved
        """
        if commit_message is None:
            commit_message = "Upload tokenizer"
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        # upload file
        operations = [
            CommitOperationAdd(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
            )
        ]

        if revision is not None:
            create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)

        print(f"Uploading the tokenizer to {repo_id}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    @classmethod
    def push_to_hub(
        cls,
        tokenizer,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
    ) -> str:
        """
        Uploads the tokenizer to the 🤗 Hub.

        :param tokenizer: `MIDITokenizer` object to save.
        :param repo_id: The name of the Hugging Face repository to push your tokenizer to. It should contain your
            organization name when pushing to a given organization.
        :param use_temp_dir: Whether to use a temporary directory to store the files saved before they are
            pushed to the Hub. Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
        :param commit_message: Message to commit while pushing. Will default to `"Upload tokenizer"`.
        :param private: Set the repository created private.
        :param token: The token to use as HTTP bearer authorization for remote files. If `True`, will use the token
            generated when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if
            `repo_url` is not specified.
        :param create_pr: Whether to create a PR with the uploaded files or directly commit (default: False).
        :param revision: Branch to push the uploaded files to. (default: None)
        :return: Instance of [`CommitInfo`] containing information about the newly created commit (commit hash, commit
                url, pr url, commit message,...).
        """
        # Repo_id is passed correctly: infer working_dir from it
        working_dir = repo_id.split("/")[-1]

        repo_id = cls._create_repo(
            repo_id,
            private=private,
            token=token,
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(
            working_dir=working_dir, use_temp_dir=use_temp_dir
        ) as work_dir:

            file_path = Path(work_dir, "tokenizer.conf")
            tokenizer.save_params(file_path)
            return cls._upload_modified_files(
                file_path,
                repo_id,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
            )

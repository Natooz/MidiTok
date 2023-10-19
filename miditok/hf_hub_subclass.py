"""
Hugging Face hub integration by subclassing transformers.utils.PushToHubMixin
Pros: simpler / less code in MidiTok, can rely on transformers updates/changes
Cons: adds a big dependency, slower load (import), no custom behavior
"""

from typing import Optional, Union
from pathlib import Path

from transformers.utils import PushToHubMixin
from transformers.utils import logging
from huggingface_hub import CommitOperationAdd, create_branch, create_commit


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MidiTokPushToHubMixin(PushToHubMixin):
    """
    Subclassed class of transformers.utils.PushToHubMixin
    """

    def _upload_modified_files(
        self,
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
        # TODO is it necessary ? What about tokenizer.push_to_hub ?
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

        logger.info(f"Uploading the tokenizer to {repo_id}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    def save_pretrained(self):
        toto = 0
        # TODO if transformers dependency --> need to create an inheriting class with save_pretrained
        # returns a list (change to one?) of the paths to the files saved

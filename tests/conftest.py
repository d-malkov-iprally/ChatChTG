import os

import pytest

from chatchtg.answering import AnswerService
from chatchtg.context.baseline import BaselineComposer
from chatchtg.documents.blobs import LocalBlobStore
from chatchtg.documents.library import DocumentLibrary
from chatchtg.documents.metadata import SqliteMetadataStore
from chatchtg.llm.fake import FakeLLM

PNG = b"\x89PNG\r\n\x1a\n" + b"fake png body"
JPEG = b"\xff\xd8\xff" + b"fake jpeg body"


@pytest.fixture(autouse=True)
def no_project_env_file(tmp_path, monkeypatch):
    """Run each test in an empty folder and without the developer's CHATCHTG_* variables.

    A .env file or a shell that selects the claude provider would otherwise make a test
    send billed requests.
    """
    monkeypatch.chdir(tmp_path)
    for name in [name for name in os.environ if name.startswith("CHATCHTG_")]:
        monkeypatch.delenv(name)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def library(tmp_path, data_root):
    return DocumentLibrary(
        blobs=LocalBlobStore(tmp_path / "var" / "blobs", data_root),
        metadata=SqliteMetadataStore(tmp_path / "var" / "chatchtg.db"),
    )


@pytest.fixture
def service(library):
    return AnswerService(library, BaselineComposer(), FakeLLM())

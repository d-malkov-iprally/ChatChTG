import pytest

from chatchtg.answering import AnswerService
from chatchtg.context.baseline import BaselineComposer
from chatchtg.documents.blobs import LocalBlobStore
from chatchtg.documents.library import DocumentLibrary
from chatchtg.documents.metadata import SqliteMetadataStore
from chatchtg.llm.fake import FakeLLM

PNG = b"\x89PNG\r\n\x1a\n" + b"fake png body"
JPEG = b"\xff\xd8\xff" + b"fake jpeg body"


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

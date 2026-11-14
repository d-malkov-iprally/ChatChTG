"""The answer service: (document, message) -> response.

The user interface and the eval runner both call this service.
"""

from __future__ import annotations

from pydantic import BaseModel

from chatchtg.context.composer import Composer, ComposerInput, composer_id
from chatchtg.context.types import Context
from chatchtg.documents.library import DocumentLibrary
from chatchtg.llm.client import Completion, LLMClient
from chatchtg.schemas import AnswerRequest, AnswerResponse


class AnswerTrace(BaseModel):
    """A response together with what produced it. For evals and debugging."""

    response: AnswerResponse
    context: Context
    completion: Completion


class AnswerService:
    def __init__(self, library: DocumentLibrary, composer: Composer, llm: LLMClient) -> None:
        self._library = library
        self._composer = composer
        self._llm = llm

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        return self.answer_traced(request).response

    def answer_traced(self, request: AnswerRequest) -> AnswerTrace:
        document = self._library.get(request.document_id)
        content = self._library.content(document)
        context = self._composer.compose(
            ComposerInput(document=document, content=content, message=request.message)
        )
        completion = self._llm.complete(context)
        response = AnswerResponse(
            text=completion.text,
            composer=composer_id(self._composer),
            model=completion.model,
        )
        return AnswerTrace(response=response, context=context, completion=completion)

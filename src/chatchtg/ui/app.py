"""Streamlit user interface. It holds no logic: it calls the library and the answer service."""

from __future__ import annotations

import json

import streamlit as st

from chatchtg.config import Settings
from chatchtg.documents.library import UnsupportedDocumentError
from chatchtg.schemas import AnswerRequest
from chatchtg.wiring import build_answer_service, build_library

st.set_page_config(page_title="ChatChTG", layout="wide")


@st.cache_resource
def services():
    settings = Settings.from_env()
    library = build_library(settings)
    return library, build_answer_service(settings, library)


library, answer_service = services()

st.title("ChatChTG")
st.caption("Ask a question about one construction drawing.")

document = None
source = st.radio("Document", ["Upload an image", "Registered document"], horizontal=True)
if source == "Upload an image":
    upload = st.file_uploader("Drawing (PNG or JPEG)")
    if upload is not None:
        try:
            document = library.add_upload(upload.getvalue(), upload.name)
        except UnsupportedDocumentError as error:
            st.error(str(error))
else:
    records = library.list(limit=200)
    if records:
        document = st.selectbox(
            "Registered documents (newest 200)",
            records,
            format_func=lambda record: f"{record.filename} ({record.id[:8]})",
        )
    else:
        st.info("No documents yet. Upload one, or run `chatchtg ingest data/preview`.")

if document is not None:
    image_column, chat_column = st.columns(2)

    with image_column:
        st.image(library.read(document.id), caption=document.filename)
        with st.expander("Document record"):
            st.json(document.model_dump(mode="json", exclude={"attributes"}))
        with st.form("attributes"):
            edited = st.text_area(
                "Attributes (JSON object)",
                value=json.dumps(document.attributes, indent=2, ensure_ascii=False),
                key=f"attributes-{document.id}",
            )
            if st.form_submit_button("Save attributes"):
                try:
                    attributes = json.loads(edited)
                    if not isinstance(attributes, dict):
                        raise ValueError("The attributes must be a JSON object.")
                except ValueError as error:
                    st.error(f"Attributes not saved: {error}")
                else:
                    removed = {key: None for key in document.attributes if key not in attributes}
                    library.patch_attributes(document.id, {**attributes, **removed})
                    st.success("Attributes saved.")

    with chat_column:
        with st.form("ask"):
            message = st.text_area("Message", placeholder="How many windows are shown?")
            submitted = st.form_submit_button("Ask")
        if submitted and message.strip():
            with st.spinner("Answering..."):
                response = answer_service.answer(
                    AnswerRequest(document_id=document.id, message=message)
                )
            st.markdown(response.text)
            st.caption(f"Composer: {response.composer} · Model: {response.model}")

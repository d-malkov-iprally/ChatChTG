"""Streamlit user interface. It holds no logic: it calls the library, the answer service and
the evidence module, which decides what is drawn on the drawing and how."""

from __future__ import annotations

import json

import streamlit as st

from chatchtg.config import Settings
from chatchtg.documents.library import UnsupportedDocumentError
from chatchtg.evidence import colour_key, describe_method, geometry, legend_rows, render_evidence
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
    try:
        content = library.read(document.id)
    except Exception as error:  # the file is gone or was replaced
        st.error(f"This drawing cannot be shown: {error}")
        st.stop()

    image_column, chat_column = st.columns(2)

    # The chat column runs first, because the image depends on the answer and on the zoom control.
    evidence = []  # of the answer that is shown, when it is shown on the drawing
    focus = None
    with chat_column:
        with st.form("ask"):
            message = st.text_area("Message", placeholder="How many windows are shown?")
            submitted = st.form_submit_button("Ask")
        if submitted and message.strip():
            with st.spinner("Answering..."):
                response = answer_service.answer(
                    AnswerRequest(document_id=document.id, message=message)
                )
            # Kept for the reruns that the controls below cause. The number gives the controls
            # of each answer their own state.
            st.session_state["answer"] = {
                "document_id": document.id,
                "message": message,
                "response": response,
                "number": st.session_state.get("answer", {}).get("number", 0) + 1,
            }
        answer = st.session_state.get("answer")
        if answer is not None and answer["document_id"] == document.id:  # never on another drawing
            response = answer["response"]
            # The form can hold a later question that was not answered.
            st.caption(f"Question: {answer['message']}")
            st.markdown(response.text)
            st.caption(f"Composer: {response.composer} · Model: {response.model}")
            st.caption(f"Method: {describe_method(response.method)}")
            if response.evidence_note:
                st.warning(response.evidence_note)
            if response.evidence:
                st.table(legend_rows(response.evidence))
                drawn = [index for index, item in enumerate(response.evidence) if geometry(item) is not None]
                if drawn and st.toggle(
                    "Show the evidence on the drawing", value=True, key=f"show-{answer['number']}"
                ):
                    evidence = response.evidence
                    focus = st.selectbox(
                        "Zoom to",
                        [None, *drawn],
                        format_func=lambda index: (
                            # The number is the row of the table: items can have the same text.
                            "Whole drawing" if index is None else f"{index}: {evidence[index].get('text', '')}"
                        ),
                        key=f"zoom-{answer['number']}",
                    )
            else:
                st.info("No evidence on the drawing for this answer.")

    with image_column:
        annotated = None
        if evidence:
            try:
                annotated = render_evidence(content, evidence, focus)
            except Exception:  # a fault in the evidence must not hide the drawing
                st.warning("The evidence could not be drawn. This is the plain drawing.")
        try:
            if annotated is not None:
                st.image(
                    annotated,
                    caption=f"Evidence on {document.filename}",
                    # A zoomed part is small: enlarge it to the column.
                    width="content" if focus is None else "stretch",
                )
                st.markdown(" · ".join(f":{colour}[■] {words}" for colour, words in colour_key(evidence)))
            else:
                st.image(content, caption=document.filename)
        except Exception as error:  # not a real image
            st.error(f"This drawing cannot be shown: {error}")
            st.stop()
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

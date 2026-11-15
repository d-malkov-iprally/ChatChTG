"""Streamlit user interface. It holds no logic: it calls the library, the answer service and
the evidence module, which decides what is drawn on the drawing and how."""

from __future__ import annotations

import html
import io
import json

import streamlit as st
from PIL import Image

from chatchtg.config import Settings
from chatchtg.demo import load_demo_set, register_demo_set
from chatchtg.documents.library import UnsupportedDocumentError
from chatchtg.evidence import colour_key, describe_method, geometry, legend_rows, render_evidence
from chatchtg.schemas import AnswerRequest
from chatchtg.wiring import build_answer_service, build_library

st.set_page_config(page_title="ChatChTG", page_icon="🧭", layout="wide")


@st.cache_resource
def services():
    settings = Settings.from_env()
    library = build_library(settings)
    demo = register_demo_set(library, settings.data_root, load_demo_set())
    return library, build_answer_service(settings, library), demo


@st.cache_data(show_spinner=False)
def thumbnail(document_id: str, _library) -> bytes | None:
    """A small copy of the drawing for its card, or None when it is not a readable image."""
    try:
        image = Image.open(io.BytesIO(_library.read(document_id))).convert("RGB")
    except Exception:
        return None
    image.thumbnail((360, 240))
    small = io.BytesIO()
    image.save(small, format="PNG")
    return small.getvalue()


def inject_styles() -> None:
    """Apply the CAD workspace visual language without changing app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --canvas: #0d111b;
            --ink: #edf1fa;
            --panel: rgba(15, 21, 33, 0.78);
            --violet: #9f7aea;
            --violet-deep: #7355c7;
        }

        [data-testid="stAppViewContainer"] {
            color: var(--ink);
            background-color: var(--canvas);
            background-image:
                linear-gradient(rgba(91, 112, 150, 0.07) 1px, transparent 1px),
                linear-gradient(90deg, rgba(91, 112, 150, 0.07) 1px, transparent 1px),
                radial-gradient(circle at 76% 0%, rgba(103, 80, 177, 0.20), transparent 28rem);
            background-size: 24px 24px, 24px 24px, auto;
        }

        [data-testid="stHeader"] { display: none; }
        [data-testid="stMainBlockContainer"] {
            max-width: none;
            padding: 0.65rem 1.1rem 1.1rem;
        }

        .top-chrome {
            display: flex;
            align-items: center;
            min-height: 3.4rem;
            margin: -0.1rem -1.1rem 0.7rem;
            padding: 0 1.4rem;
            color: #f8f9ff;
            background: linear-gradient(100deg, #111725 0%, #121827 66%, #161b2c 100%);
            border-bottom: 1px solid rgba(170, 184, 219, 0.22);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.22);
        }
        .brand-mark { display: flex; align-items: center; gap: 0.68rem; min-width: 12.5rem; }
        .brand-compass { width: 1.55rem; height: 1.55rem; color: #c7afff; filter: drop-shadow(0 2px 5px rgba(161, 122, 234, 0.32)); }
        .brand-compass path, .brand-compass circle { vector-effect: non-scaling-stroke; }
        .brand-name { font-size: 1.14rem; font-weight: 740; letter-spacing: -0.02em; }
        div[data-testid="stSegmentedControl"] { margin: 0.05rem 0 0.6rem; }
        div[data-testid="stSegmentedControl"] label {
            color: #edf1fa !important; background: rgba(15, 21, 33, 0.78) !important;
            border-color: rgba(181, 192, 218, 0.18) !important;
        }
        div[data-testid="stSegmentedControl"] label p { color: inherit !important; }
        div[data-testid="stSegmentedControl"] label:has(input:checked) {
            color: #cdbbff !important; background: rgba(115, 85, 199, 0.26) !important;
            border-color: #a98be9 !important; box-shadow: 0 0 0 1px rgba(188, 165, 251, 0.22);
        }
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] label p { color: #e9edf7 !important; }

        .drawing-tab {
            display: inline-flex; align-items: center; gap: 0.75rem; min-height: 2rem; margin: 0.08rem 0 0.42rem 0.7rem;
            padding: 0 0.75rem; color: #eef1f9; background: linear-gradient(180deg, #22293a, #161c2a);
            border: 1px solid rgba(193, 204, 232, 0.27); border-bottom: 0; border-radius: 0.5rem 0.5rem 0 0;
            font-size: 0.88rem; font-weight: 650;
        }

        .st-key-drawing_canvas {
            padding: 0.8rem 0.8rem 0.55rem; border: 1px solid rgba(174, 189, 223, 0.64);
            border-radius: 0.5rem; background-color: #101722;
            background-image:
                linear-gradient(rgba(82, 117, 145, 0.13) 1px, transparent 1px),
                linear-gradient(90deg, rgba(82, 117, 145, 0.13) 1px, transparent 1px),
                linear-gradient(rgba(82, 117, 145, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(82, 117, 145, 0.05) 1px, transparent 1px);
            background-size: 24px 24px, 24px 24px, 6px 6px, 6px 6px;
            box-shadow: 0 13px 32px rgba(0, 0, 0, 0.2);
        }
        .canvas-caption { display: flex; align-items: center; justify-content: space-between; margin: 0 0.2rem 0.65rem; color: #c1ccdc; font-size: 0.78rem; }
        .canvas-caption strong { color: #f5f7fc; font-weight: 650; }
        .st-key-drawing_canvas [data-testid="stImage"] { padding: 0.85rem; background: #fff; border: 1px solid rgba(220, 226, 237, 0.9); box-shadow: 0 16px 28px rgba(0, 0, 0, 0.28); }
        .st-key-drawing_canvas [data-testid="stImage"] img { background: #fff; object-fit: contain; max-height: 26rem; }
        .canvas-status { display: flex; gap: 1rem; margin: 0.7rem 0.2rem 0.05rem; color: #bdc7d8; font-size: 0.72rem; }
        .canvas-status span::before { content: ""; display: inline-block; width: 0.43rem; height: 0.43rem; margin-right: 0.35rem; border-radius: 50%; background: #69d2a4; }
        .st-key-drawing_canvas [data-testid="stExpander"] {
            color: #e9edf7; background: rgba(11, 17, 28, 0.62); border-color: rgba(187, 200, 229, 0.62);
        }
        .st-key-drawing_canvas [data-testid="stExpander"] summary,
        .st-key-drawing_canvas [data-testid="stExpander"] summary p { color: #e9edf7 !important; }

        .st-key-assistant_panel {
            padding: 0.95rem; border: 1px solid rgba(174, 189, 223, 0.64); border-radius: 0.5rem;
            background: var(--panel); box-shadow: 0 13px 32px rgba(0, 0, 0, 0.17);
        }
        .assistant-heading { display: flex; align-items: center; gap: 0.52rem; margin: 0.1rem 0 0.8rem; padding-bottom: 0.76rem; color: #f5f7fc; border-bottom: 1px solid rgba(174, 189, 223, 0.3); font-size: 1.03rem; font-weight: 760; }
        .assistant-spark { color: #c7afff; font-size: 1.3rem; }
        .assistant-intro { margin: 0.3rem 0 1rem; color: #bdc7d8; font-size: 0.88rem; line-height: 1.45; }
        .question-label { color: #bdc7d8; font-size: 0.7rem; font-weight: 760; letter-spacing: 0.09em; text-transform: uppercase; }
        .assistant-question { margin: 0.35rem 0 0.7rem; padding: 0.66rem 0.75rem; color: #f0f3fa; background: rgba(65, 75, 101, 0.34); border: 1px solid rgba(174, 189, 223, 0.3); border-radius: 0.55rem 0.55rem 0.18rem 0.55rem; font-size: 0.9rem; line-height: 1.4; }
        .assistant-answer-label { margin: 0.7rem 0 0.32rem; color: #cdbbff; font-size: 0.7rem; font-weight: 760; letter-spacing: 0.09em; text-transform: uppercase; }
        .st-key-assistant_panel div[data-testid="stTextArea"] textarea { min-height: 5.2rem; color: #f3f5fb; background: rgba(8, 13, 23, 0.66); border: 1px solid rgba(174, 189, 223, 0.4); border-radius: 0.48rem; }
        .st-key-assistant_panel div[data-testid="stTextArea"] textarea::placeholder { color: #9ca9bf; }
        .st-key-assistant_panel div[data-testid="stTextArea"] textarea:focus { border-color: #a98be9; box-shadow: 0 0 0 1px #a98be9; }
        .st-key-assistant_panel div[data-testid="stFormSubmitButton"] button { width: 100%; color: #cdbbff; background: rgba(115, 85, 199, 0.24); border: 1px solid #a98be9; border-radius: 0.4rem; font-weight: 690; }
        .st-key-assistant_panel [data-testid="stCaptionContainer"] { color: #aebbd0; font-size: 0.7rem; }
        .st-key-assistant_panel .stMarkdown { color: #edf1fa; font-size: 0.91rem; line-height: 1.55; }

        .demo-heading { margin: 0.1rem 0 0.55rem; color: #f5d98a; font-size: 0.8rem; font-weight: 740; letter-spacing: 0.06em; text-transform: uppercase; }
        .demo-heading span { margin-left: 0.6rem; color: #aebbd0; font-weight: 500; letter-spacing: 0; text-transform: none; }
        .st-key-demo_strip { margin-bottom: 0.7rem; padding: 0.7rem 0.8rem 0.5rem; border: 1px solid rgba(245, 217, 138, 0.5); border-radius: 0.5rem; background: rgba(15, 21, 33, 0.78); }
        .st-key-demo_strip [data-testid="stImage"] { padding: 0.25rem; background: #fff; border-radius: 0.3rem; }
        .st-key-demo_strip [data-testid="stImage"] img { height: 5.2rem; object-fit: contain; background: #fff; }
        .st-key-demo_strip button { min-height: 2.9rem; color: #edf1fa; background: rgba(8, 13, 23, 0.5); border: 1px solid rgba(174, 189, 223, 0.36); border-radius: 0.4rem; }
        .st-key-demo_strip button p { font-size: 0.74rem; line-height: 1.2; }
        .st-key-demo_strip button[kind="primary"] { color: #1a1405; background: #f5d98a; border-color: #f5d98a; font-weight: 700; }
        .demo-badge { margin-right: 0.5rem; padding: 0.08rem 0.42rem; color: #1a1405; background: #f5d98a; border-radius: 0.3rem; font-size: 0.7rem; font-weight: 760; }
        .st-key-question_chips { margin: 0.35rem 0 0.8rem; gap: 0.35rem; }
        .st-key-question_chips button { min-height: 1.8rem; padding: 0.1rem 0.7rem; color: #f5d98a; background: rgba(245, 217, 138, 0.08); border: 1px solid rgba(245, 217, 138, 0.55); border-radius: 1rem; }
        .st-key-question_chips button p { font-size: 0.78rem; }
        .st-key-question_chips button[kind="primary"] { color: #9fe3c2; background: rgba(105, 210, 164, 0.12); border-color: rgba(105, 210, 164, 0.6); }

        .empty-state { padding: 2.6rem 1.5rem; text-align: center; color: #dce3f2; border: 1px dashed rgba(183, 196, 226, 0.42); border-radius: 0.5rem; background: rgba(13, 19, 30, 0.5); }
        .empty-state strong { display: block; margin-bottom: 0.45rem; color: #fff; font-size: 1.05rem; }
        .empty-state span { color: #aebbd0; font-size: 0.9rem; }
        [data-testid="stFileUploader"] { padding: 0.3rem; color: #edf1fa; background: rgba(15, 21, 33, 0.78); border: 1px solid rgba(174, 189, 223, 0.5); border-radius: 0.48rem; }
        [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] label p,
        [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #edf1fa !important; }
        [data-testid="stFileUploader"] section { background: rgba(8, 13, 23, 0.48) !important; border-color: rgba(174, 189, 223, 0.36) !important; }
        [data-testid="stFileUploader"] button { color: #cdbbff !important; background: rgba(115, 85, 199, 0.2) !important; border-color: #a98be9 !important; }
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { color: #edf1fa !important; background: rgba(15, 21, 33, 0.78) !important; border-color: rgba(174, 189, 223, 0.5) !important; border-radius: 0.42rem; }
        [data-testid="stSelectbox"] svg { fill: #cdbbff !important; }
        div[data-baseweb="popover"] ul { color: #edf1fa !important; background: #161d2b !important; }
        div[data-baseweb="popover"] li { color: #edf1fa !important; }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] { padding: 0.4rem 0.55rem 0.7rem; }
            .top-chrome { margin-left: -0.55rem; margin-right: -0.55rem; padding: 0 0.7rem; }
            .st-key-drawing_canvas [data-testid="stImage"] img { max-height: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="top-chrome">
          <div class="brand-mark">
            <svg class="brand-compass" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="4.5" r="1.8"></circle><path d="M10.8 5.9 5.9 20"></path><path d="m13.2 5.9 4.9 14.1"></path><path d="M8.3 13.2h7.4"></path><path d="m5.9 20 2.3-2.2"></path><path d="m18.1 20-2.3-2.2"></path>
            </svg>
            <span class="brand-name">ChatChTG</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


DEMO, UPLOAD, REGISTERED = "★ Demo drawings", "Upload an image", "Registered document"


def choose_demo_drawing(index: int) -> None:
    st.session_state["demo_drawing"] = index


def render_demo_picker(library, demo):
    """The hand-picked drawings as a row of cards. Returns the chosen drawing and its record."""
    chosen = min(st.session_state.get("demo_drawing", 0), len(demo) - 1)
    with st.container(key="demo_strip", border=False):
        st.markdown(
            '<div class="demo-heading">★ Suggested'
            "<span>Each drawing has ready questions with the benchmark's answer to compare with.</span></div>",
            unsafe_allow_html=True,
        )
        for index, column in enumerate(st.columns(len(demo), gap="small")):
            drawing, record = demo[index]
            with column:
                small = thumbnail(record.id, library)
                if small is not None:
                    st.image(small, width="stretch")
                st.button(
                    f"{index + 1} · {drawing.title}",
                    key=f"demo-{index}",
                    help=drawing.note or None,
                    type="primary" if index == chosen else "secondary",
                    width="stretch",
                    on_click=choose_demo_drawing,
                    args=(index,),
                )
    drawing, record = demo[chosen]
    return drawing, record


def render_document_picker(library, demo):
    """Returns the open document and, for a drawing of the demo set, its entry."""
    options = [DEMO, UPLOAD, REGISTERED] if demo else [UPLOAD, REGISTERED]
    source = st.segmented_control("Document", options, default=options[0], label_visibility="collapsed")
    # A click on the selected option clears the control: that is still the first option.
    if source is None:
        source = options[0]
    if source == DEMO:
        drawing, record = render_demo_picker(library, demo)
        return record, drawing
    by_id = {record.id: drawing for drawing, record in demo}
    if source == UPLOAD:
        upload = st.file_uploader("Drawing (PNG or JPEG)")
        if upload is None:
            return None, None
        try:
            record = library.add_upload(upload.getvalue(), upload.name)
        except UnsupportedDocumentError as error:
            st.error(str(error))
            return None, None
        return record, by_id.get(record.id)

    records = library.list(limit=200)
    if records:
        record = st.selectbox(
            "Registered documents (newest 200)",
            records,
            format_func=lambda record: (
                f"{'★ ' if record.id in by_id else ''}{record.filename} ({record.id[:8]})"
            ),
        )
        return record, by_id.get(record.id)
    st.info("No documents yet. Upload one, or run `chatchtg ingest data/preview`.")
    return None, None


def ask_demo_question(document_id: str, index: int) -> None:
    st.session_state["demo_question"] = (document_id, index)


def keep_answer(answer_service, document, message: str, reference: str | None) -> None:
    """Answer the question and keep the answer for the reruns that the controls cause."""
    with st.spinner("Analyzing drawing..."):
        response = answer_service.answer(AnswerRequest(document_id=document.id, message=message))
    st.session_state.setdefault("asked", set()).add((document.id, message))
    # The number gives the controls of each answer their own state.
    st.session_state["answer"] = {
        "document_id": document.id,
        "message": message,
        "response": response,
        "reference": reference,
        "number": st.session_state.get("answer", {}).get("number", 0) + 1,
    }


def render_workspace(document, library, answer_service, demo_drawing=None) -> None:
    try:
        content = library.read(document.id)
    except Exception as error:  # the file is gone or was replaced
        st.error(f"This drawing cannot be shown: {error}")
        st.stop()

    filename = html.escape(document.filename)
    badge = '<span class="demo-badge">★ DEMO</span>' if demo_drawing is not None else ""
    title = f" · {html.escape(demo_drawing.title)}" if demo_drawing is not None else ""
    st.markdown(f'<div class="drawing-tab">{badge}{filename}{title}</div>', unsafe_allow_html=True)
    questions = demo_drawing.questions if demo_drawing is not None else ()
    image_column, assistant_column = st.columns((2.5, 1), gap="small")

    # The assistant column runs first, because the image depends on the answer and on the zoom control.
    evidence = []  # of the answer that is shown, when it is shown on the drawing
    focus = None
    with assistant_column:
        with st.container(key="assistant_panel", border=False):
            st.markdown(
                '<div class="assistant-heading"><span class="assistant-spark">✦</span>AI Assistant</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="assistant-intro">Ask a precise question about the open drawing. The answer stays connected to this sheet.</p>',
                unsafe_allow_html=True,
            )
            pending = st.session_state.pop("demo_question", None)
            if pending is not None and pending[0] == document.id and pending[1] < len(questions):
                question = questions[pending[1]]
                keep_answer(answer_service, document, question.text, question.reference)
            if questions:
                # The trail of the demo: a click asks the question, a question that was asked changes colour.
                asked = st.session_state.get("asked", set())
                st.markdown('<div class="question-label">Demo questions</div>', unsafe_allow_html=True)
                with st.container(key="question_chips", horizontal=True, border=False):
                    for index, question in enumerate(questions):
                        done = (document.id, question.text) in asked
                        st.button(
                            f"{index + 1} · {question.label}",
                            key=f"chip-{document.id[:12]}-{index}",
                            help=question.text,
                            type="primary" if done else "secondary",
                            on_click=ask_demo_question,
                            args=(document.id, index),
                        )
            with st.form("ask"):
                message = st.text_area(
                    "Ask about this drawing",
                    placeholder="How many doors are in this floor plan?",
                )
                submitted = st.form_submit_button("Analyze drawing")

            if submitted and message.strip():
                reference = next((q.reference for q in questions if q.text == message.strip()), None)
                keep_answer(answer_service, document, message, reference)
            answer = st.session_state.get("answer")
            if answer is not None and answer["document_id"] == document.id:  # never on another drawing
                response = answer["response"]
                # The form can hold a later question that was not answered.
                st.markdown('<div class="question-label">Your question</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="assistant-question">{html.escape(answer["message"])}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="assistant-answer-label">Drawing answer</div>', unsafe_allow_html=True)
                st.markdown(response.text)
                st.caption(f"Composer: {response.composer} · Model: {response.model}")
                st.caption(f"Method: {describe_method(response.method)}")
                if answer.get("reference"):
                    st.caption(f"Benchmark answer: {answer['reference']}")
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
        with st.container(key="drawing_canvas", border=False):
            st.markdown(
                f'<div class="canvas-caption"><strong>{filename}</strong></div>',
                unsafe_allow_html=True,
            )
            annotated = None
            if evidence:
                try:
                    annotated = render_evidence(content, evidence, focus)
                except Exception:  # a fault in the evidence must not hide the drawing
                    st.warning("The evidence could not be drawn. This is the plain drawing.")
            try:
                if annotated is not None:
                    st.image(annotated, caption=f"Evidence on {document.filename}", width="stretch")
                    st.markdown(" · ".join(f":{colour}[■] {words}" for colour, words in colour_key(evidence)))
                else:
                    st.image(content, width="stretch")
            except Exception as error:  # not a real image
                st.error(f"This drawing cannot be shown: {error}")
                st.stop()
            st.markdown(
                '<div class="canvas-status"><span>Drawing ready</span><span>White paper view</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Drawing details"):
                st.json(document.model_dump(mode="json", exclude={"attributes"}))
                with st.form("attributes"):
                    edited = st.text_area(
                        "Attributes (JSON object)",
                        value=json.dumps(document.attributes, indent=2, ensure_ascii=False),
                        key=f"attributes-{document.id}",
                    )
                    if st.form_submit_button("Save drawing details"):
                        try:
                            attributes = json.loads(edited)
                            if not isinstance(attributes, dict):
                                raise ValueError("The attributes must be a JSON object.")
                        except ValueError as error:
                            st.error(f"Details not saved: {error}")
                        else:
                            removed = {key: None for key in document.attributes if key not in attributes}
                            library.patch_attributes(document.id, {**attributes, **removed})
                            st.success("Drawing details saved.")


def render_empty_workspace() -> None:
    with st.container(key="drawing_canvas", border=False):
        st.markdown(
            '<div class="empty-state"><strong>Open a construction drawing</strong><span>Upload a PNG or JPEG, or select a registered drawing to begin analysis.</span></div>',
            unsafe_allow_html=True,
        )


inject_styles()
library, answer_service, demo = services()
render_header()
document, demo_drawing = render_document_picker(library, demo)

if document is None:
    render_empty_workspace()
else:
    render_workspace(document, library, answer_service, demo_drawing)

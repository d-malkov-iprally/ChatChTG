"""Streamlit user interface. It holds no logic: it calls the library and the answer service."""

from __future__ import annotations

import html
import json

import streamlit as st

from chatchtg.config import Settings
from chatchtg.documents.library import UnsupportedDocumentError
from chatchtg.schemas import AnswerRequest
from chatchtg.wiring import build_answer_service, build_library

st.set_page_config(page_title="ChatChTG", page_icon="🧭", layout="wide")


@st.cache_resource
def services():
    settings = Settings.from_env()
    library = build_library(settings)
    return library, build_answer_service(settings, library)


def inject_styles() -> None:
    """Apply the CAD workspace visual language without changing app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --canvas: #0d111b;
            --ink: #172033;
            --panel: rgba(251, 252, 255, 0.97);
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
        .primary-nav { display: flex; gap: 0.36rem; height: 100%; align-items: center; }
        .primary-nav span { color: #c5cad9; padding: 0.55rem 1rem; border-radius: 0.35rem; font-size: 0.94rem; }
        .primary-nav .active { color: #fff; background: linear-gradient(135deg, #62558f, #413d65); box-shadow: inset 0 0 0 1px rgba(222, 210, 255, 0.15); }
        div[data-testid="stSegmentedControl"] { margin: 0.05rem 0 0.6rem; }
        div[data-testid="stSegmentedControl"] button {
            color: #edf1fa; background: rgba(15, 21, 33, 0.78); border-color: rgba(181, 192, 218, 0.18);
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            background: rgba(115, 85, 199, 0.62); border-color: rgba(191, 167, 255, 0.7);
        }
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] label p { color: #e9edf7 !important; }

        .drawing-tab {
            display: inline-flex; align-items: center; gap: 0.75rem; min-height: 2rem; margin: 0.08rem 0 0.42rem 0.7rem;
            padding: 0 0.75rem; color: #eef1f9; background: linear-gradient(180deg, #22293a, #161c2a);
            border: 1px solid rgba(193, 204, 232, 0.27); border-bottom: 0; border-radius: 0.5rem 0.5rem 0 0;
            font-size: 0.88rem; font-weight: 650;
        }
        .drawing-tab .close { color: #aeb8cd; font-size: 1.05rem; }
        .drawing-tab .add { color: #d3c3ff; margin-left: 0.25rem; font-size: 1.25rem; font-weight: 400; }

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
        .canvas-coordinates { color: #9b7ee3; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.7rem; }
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
            padding: 0.95rem; border: 1px solid rgba(207, 213, 228, 0.92); border-radius: 0.5rem;
            background: var(--panel); box-shadow: 0 13px 32px rgba(0, 0, 0, 0.17);
        }
        .assistant-heading { display: flex; align-items: center; gap: 0.52rem; margin: 0.1rem 0 0.8rem; padding-bottom: 0.76rem; color: #202943; border-bottom: 1px solid #e2e5ee; font-size: 1.03rem; font-weight: 760; }
        .assistant-spark { color: var(--violet-deep); font-size: 1.3rem; }
        .assistant-intro { margin: 0.3rem 0 1rem; color: #6b768a; font-size: 0.88rem; line-height: 1.45; }
        .question-label { color: #647086; font-size: 0.7rem; font-weight: 760; letter-spacing: 0.09em; text-transform: uppercase; }
        .assistant-question { margin: 0.35rem 0 0.7rem; padding: 0.66rem 0.75rem; color: #343d54; background: #f0f2f8; border: 1px solid #e0e4ee; border-radius: 0.55rem 0.55rem 0.18rem 0.55rem; font-size: 0.9rem; line-height: 1.4; }
        .assistant-answer-label { margin: 0.7rem 0 0.32rem; color: #5f4aa5; font-size: 0.7rem; font-weight: 760; letter-spacing: 0.09em; text-transform: uppercase; }
        .st-key-assistant_panel div[data-testid="stTextArea"] textarea { min-height: 5.2rem; color: #202a3d; background: #f5f7fb; border: 1px solid #dce2ef; border-radius: 0.48rem; }
        .st-key-assistant_panel div[data-testid="stTextArea"] textarea:focus { border-color: #9a82df; box-shadow: 0 0 0 1px #9a82df; }
        .st-key-assistant_panel div[data-testid="stFormSubmitButton"] button { width: 100%; color: #fff; background: linear-gradient(135deg, #7d61c8, #5c63bd); border: 0; border-radius: 0.4rem; font-weight: 690; }
        .st-key-assistant_panel [data-testid="stCaptionContainer"] { color: #727c91; font-size: 0.7rem; }
        .st-key-assistant_panel .stMarkdown { color: #273148; font-size: 0.91rem; line-height: 1.55; }

        .empty-state { padding: 2.6rem 1.5rem; text-align: center; color: #dce3f2; border: 1px dashed rgba(183, 196, 226, 0.42); border-radius: 0.5rem; background: rgba(13, 19, 30, 0.5); }
        .empty-state strong { display: block; margin-bottom: 0.45rem; color: #fff; font-size: 1.05rem; }
        .empty-state span { color: #aebbd0; font-size: 0.9rem; }
        [data-testid="stFileUploader"] { padding: 0.3rem; background: rgba(255,255,255,0.96); border-radius: 0.48rem; }
        [data-testid="stSelectbox"] > div > div { background: #fff; border-radius: 0.42rem; }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] { padding: 0.4rem 0.55rem 0.7rem; }
            .top-chrome { margin-left: -0.55rem; margin-right: -0.55rem; padding: 0 0.7rem; }
            .primary-nav { display: none; }
            .window-actions { display: none; }
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
          <div class="primary-nav"><span class="active">Analyze</span><span>Documents</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_picker(library):
    source = st.segmented_control(
        "Document",
        ["Upload an image", "Registered document"],
        default="Upload an image",
        label_visibility="collapsed",
    )
    if source == "Upload an image":
        upload = st.file_uploader("Drawing (PNG or JPEG)")
        if upload is None:
            return None
        try:
            return library.add_upload(upload.getvalue(), upload.name)
        except UnsupportedDocumentError as error:
            st.error(str(error))
            return None

    records = library.list(limit=200)
    if records:
        return st.selectbox(
            "Registered documents (newest 200)",
            records,
            format_func=lambda record: f"{record.filename} ({record.id[:8]})",
        )
    st.info("No documents yet. Upload one, or run `chatchtg ingest data/preview`.")
    return None


def render_workspace(document, library, answer_service) -> None:
    filename = html.escape(document.filename)
    st.markdown(
        f'<div class="drawing-tab">{filename}<span class="close">×</span><span class="add">+</span></div>',
        unsafe_allow_html=True,
    )
    image_column, assistant_column = st.columns((2.5, 1), gap="small")

    with image_column:
        with st.container(key="drawing_canvas", border=False):
            st.markdown(
                f'<div class="canvas-caption"><strong>{filename}</strong><span class="canvas-coordinates">MODEL · 1:100 · 0,0</span></div>',
                unsafe_allow_html=True,
            )
            st.image(library.read(document.id), width="stretch")
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
            with st.form("ask"):
                message = st.text_area(
                    "Ask about this drawing",
                    placeholder="How many doors are in this floor plan?",
                )
                submitted = st.form_submit_button("Analyze drawing")

            if submitted and message.strip():
                with st.spinner("Analyzing drawing..."):
                    response = answer_service.answer(
                        AnswerRequest(document_id=document.id, message=message)
                    )
                st.markdown('<div class="question-label">Your question</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="assistant-question">{html.escape(message)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="assistant-answer-label">Drawing answer</div>', unsafe_allow_html=True)
                st.markdown(response.text)
                st.caption(f"Composer: {response.composer} · Model: {response.model}")

def render_empty_workspace() -> None:
    with st.container(key="drawing_canvas", border=False):
        st.markdown(
            '<div class="empty-state"><strong>Open a construction drawing</strong><span>Upload a PNG or JPEG, or select a registered drawing to begin analysis.</span></div>',
            unsafe_allow_html=True,
        )


inject_styles()
library, answer_service = services()
render_header()
document = render_document_picker(library)

if document is None:
    render_empty_workspace()
else:
    render_workspace(document, library, answer_service)

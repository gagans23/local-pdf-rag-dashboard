from pathlib import Path
from datetime import datetime
import csv
import re
from tempfile import NamedTemporaryFile

import chromadb
import ollama
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


APP_TITLE = "Local PDF RAG Intelligence Dashboard"
CHROMA_PATH = "chroma_db"
EVAL_LOG_PATH = Path("eval_log.csv")
COLLECTION_NAME = "pdf_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
]
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODELS = [
    "llama3.1:8b",
    "qwen3:32b",
    "deepseek-r1:8b",
    "gpt-oss:20b",
    "qwen2.5:14b",
    "gemma4:latest",
]
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


@st.cache_resource
def get_embedder(embedding_model):
    return SentenceTransformer(embedding_model)


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def collection_name_for_model(embedding_model):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", embedding_model).strip("_").lower()
    return f"{COLLECTION_NAME}_{slug}"[:63]


@st.cache_resource
def get_collection(embedding_model):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name_for_model(embedding_model),
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
    )


def get_collection_count(collection_name):
    try:
        return get_chroma_client().get_collection(collection_name).count()
    except Exception:
        return 0


def encode_texts(texts, is_query=False):
    embedding_model = st.session_state.embedding_model
    embedder = get_embedder(embedding_model)
    payload = texts
    if embedding_model.startswith("BAAI/bge") and is_query:
        payload = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
    return embedder.encode(payload, normalize_embeddings=True).tolist()


def upgrade_legacy_index():
    legacy_collection = get_chroma_client().get_collection(COLLECTION_NAME)
    target_collection = get_collection(st.session_state.embedding_model)
    results = legacy_collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    if not documents:
        return 0

    upgraded_metadatas = []
    for metadata in metadatas:
        upgraded_metadata = dict(metadata)
        upgraded_metadata["embedding_model"] = st.session_state.embedding_model
        upgraded_metadatas.append(upgraded_metadata)

    target_collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=encode_texts(documents),
        metadatas=upgraded_metadatas,
    )
    return len(documents)


def extract_pdf_text(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        reader = PdfReader(str(tmp_path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((page_number, text))
        return pages
    finally:
        tmp_path.unlink(missing_ok=True)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks


def index_pdfs(uploaded_files):
    collection = get_collection(st.session_state.embedding_model)

    documents = []
    metadatas = []
    ids = []

    for uploaded_file in uploaded_files:
        pages = extract_pdf_text(uploaded_file)
        for page_number, page_text in pages:
            for chunk_index, chunk in enumerate(chunk_text(page_text), start=1):
                doc_id = f"{uploaded_file.name}:p{page_number}:c{chunk_index}"
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": uploaded_file.name,
                        "page": page_number,
                        "chunk": chunk_index,
                        "embedding_model": st.session_state.embedding_model,
                    }
                )
                ids.append(doc_id)

    if not documents:
        return 0

    embeddings = encode_texts(documents)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(documents)


def get_knowledge_base_rows():
    collection = get_collection(st.session_state.embedding_model)
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    rows = []
    for metadata in results.get("metadatas", []):
        rows.append(
            {
                "source": metadata["source"],
                "page": int(metadata["page"]),
                "chunk": int(metadata["chunk"]),
            }
        )
    return rows


def get_knowledge_base_frame():
    rows = get_knowledge_base_rows()
    if not rows:
        return pd.DataFrame(columns=["source", "page", "chunk"])
    return pd.DataFrame(rows).sort_values(["source", "page", "chunk"])


def get_source_summary(kb_df):
    if kb_df.empty:
        return pd.DataFrame(columns=["source", "pages", "chunks"])

    return (
        kb_df.groupby("source")
        .agg(pages=("page", "nunique"), chunks=("chunk", "count"))
        .reset_index()
        .sort_values("chunks", ascending=False)
    )


def get_page_summary(kb_df):
    if kb_df.empty:
        return pd.DataFrame(columns=["source", "page", "chunks"])

    return (
        kb_df.groupby(["source", "page"])
        .size()
        .reset_index(name="chunks")
        .sort_values(["source", "page"])
    )


def get_retrieval_frame():
    rows = []
    for answer_index, message in enumerate(st.session_state.messages, start=1):
        if message["role"] != "assistant":
            continue
        for row in message.get("source_rows", []):
            rows.append(
                {
                    "answer": answer_index,
                    "rank": row["rank"],
                    "source": row["source"],
                    "page": row["page"],
                    "chunk": row["chunk"],
                    "distance": row["distance"],
                    "embedding_model": row.get("embedding_model", "unknown"),
                    "preview": row["preview"],
                }
            )
    return pd.DataFrame(rows)


def evaluate_retrieval(question, source_rows):
    if not source_rows:
        return {
            "question": question,
            "embedding_model": st.session_state.embedding_model,
            "confidence": 0,
            "grade": "No retrieval",
            "best_distance": None,
            "avg_distance": None,
            "source_diversity": 0,
            "page_diversity": 0,
            "citation_count": 0,
        }

    distances = [row["distance"] for row in source_rows]
    avg_distance = sum(distances) / len(distances)
    best_distance = min(distances)
    source_diversity = len({row["source"] for row in source_rows})
    page_diversity = len({(row["source"], row["page"]) for row in source_rows})
    confidence = round(max(0, min(100, (1 - avg_distance) * 100)))

    if confidence >= 75:
        grade = "Strong"
    elif confidence >= 50:
        grade = "Moderate"
    elif confidence > 0:
        grade = "Weak"
    else:
        grade = "Poor"

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "embedding_model": st.session_state.embedding_model,
        "confidence": confidence,
        "grade": grade,
        "best_distance": round(best_distance, 4),
        "avg_distance": round(avg_distance, 4),
        "source_diversity": source_diversity,
        "page_diversity": page_diversity,
        "citation_count": len(source_rows),
    }


def get_eval_frame():
    if not EVAL_LOG_PATH.exists():
        return pd.DataFrame()
    eval_df = pd.read_csv(EVAL_LOG_PATH, escapechar="\\", on_bad_lines="skip")
    if "embedding_model" in eval_df:
        eval_df = eval_df[
            eval_df["embedding_model"].astype(str).str.contains("/", regex=False, na=False)
        ].copy()
    numeric_columns = [
        "confidence",
        "best_distance",
        "avg_distance",
        "source_diversity",
        "page_diversity",
        "citation_count",
    ]
    for column in numeric_columns:
        if column in eval_df:
            eval_df[column] = pd.to_numeric(eval_df[column], errors="coerce")
    if "confidence" in eval_df:
        eval_df["confidence"] = eval_df["confidence"].fillna(0)
    return eval_df


def persist_eval(eval_row):
    pd.DataFrame([eval_row]).to_csv(
        EVAL_LOG_PATH,
        mode="a",
        header=not EVAL_LOG_PATH.exists(),
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )


def apply_dashboard_style():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.25rem;
            max-width: 1280px;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 14px;
            margin: 0.35rem 0 1.25rem;
        }
        .kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f7f9fc 100%);
            border: 1px solid #e7edf5;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 10px 26px rgba(22, 34, 51, 0.06);
        }
        .kpi-label {
            color: #667085;
            font-size: 0.82rem;
            margin-bottom: 8px;
            white-space: nowrap;
        }
        .kpi-value {
            color: #111827;
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.05;
        }
        .kpi-caption {
            color: #667085;
            font-size: 0.72rem;
            margin-top: 8px;
        }
        .impact-strip {
            border: 1px solid #d8e2ef;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            padding: 16px 18px;
            background: #f8fbff;
            margin: 10px 0 22px 0;
        }
        .impact-strip h4 {
            margin: 0 0 6px 0;
            color: #111827;
        }
        .impact-strip p {
            margin: 0;
            color: #475467;
        }
        .section-kicker {
            color: #2563eb;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: -0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=46, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_impact_statement(kb_df, source_summary, retrieval_df):
    if kb_df.empty:
        st.info("Upload and index a PDF to turn this into an interactive document intelligence dashboard.")
        return

    pages = kb_df["page"].nunique()
    chunks = len(kb_df)
    docs = len(source_summary)
    retrieval_count = len(retrieval_df)
    evidence_sentence = (
        f" Current chat evidence includes {retrieval_count} retrieved chunks."
        if retrieval_count
        else ""
    )
    st.markdown(
        f"""
        <div class="impact-strip">
            <h4>Operational impact</h4>
            <p>
                This knowledge base converts {docs} document(s), {pages} page(s), and {chunks} semantic chunks
                into a searchable decision layer. Retrieval evidence is tracked for every answer, so users can
                see which pages are influencing model output.{evidence_sentence}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_enterprise_metrics(kb_df, source_summary, retrieval_df, eval_df):
    chunks = len(kb_df)
    pages = 0 if kb_df.empty else kb_df["page"].nunique()
    docs = 0 if source_summary.empty else len(source_summary)
    avg_chunks_per_page = 0 if pages == 0 else chunks / pages
    avg_confidence = None if eval_df.empty else eval_df["confidence"].mean()
    health = "No evals" if avg_confidence is None else f"{avg_confidence:.0f}%"
    health_caption = "Ask a question to score retrieval" if avg_confidence is None else "Average confidence"

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Knowledge chunks</div>
                <div class="kpi-value">{chunks:,}</div>
                <div class="kpi-caption">Searchable semantic units</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Documents</div>
                <div class="kpi-value">{docs:,}</div>
                <div class="kpi-caption">Indexed sources</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Pages covered</div>
                <div class="kpi-value">{pages:,}</div>
                <div class="kpi-caption">Source footprint</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg chunks/page</div>
                <div class="kpi-value">{avg_chunks_per_page:.1f}</div>
                <div class="kpi-caption">Information density</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Eval health</div>
                <div class="kpi-value">{health}</div>
                <div class="kpi-caption">{health_caption}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_command_center(kb_df, source_summary, retrieval_df):
    if kb_df.empty:
        return

    chunks = len(kb_df)
    pages = kb_df["page"].nunique()
    docs = len(source_summary)
    readiness = min(
        100,
        round((min(chunks, 100) * 0.45) + (min(pages, 50) * 0.6) + (20 if docs else 0)),
    )

    left_col, right_col = st.columns([0.9, 1.1])
    with left_col:
        st.markdown(
            f"""
            <div class="impact-strip">
                <h4>Decision readiness</h4>
                <p>
                    The system has enough indexed context to support document Q&A across {pages} pages.
                    Evidence traceability is active; once users ask questions, retrieval quality will be scored
                    and surfaced as answer evidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=readiness,
                number={"suffix": "%", "font": {"size": 46}},
                delta={"reference": 75, "increasing": {"color": "#0f766e"}},
                title={"text": "Knowledge readiness", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                    "bar": {"color": "#2563eb", "thickness": 0.22},
                    "bgcolor": "#f8fafc",
                    "borderwidth": 1,
                    "bordercolor": "#d8e2ef",
                    "steps": [
                        {"range": [0, 40], "color": "#fee2e2"},
                        {"range": [40, 75], "color": "#fef3c7"},
                        {"range": [75, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#111827", "width": 3},
                        "thickness": 0.75,
                        "value": 75,
                    },
                },
            )
        )
        fig.update_layout(
            title="Enterprise readiness score",
            showlegend=False,
        )
        st.plotly_chart(plotly_layout(fig, 300), use_container_width=True, config=PLOTLY_CONFIG)


def render_coverage_charts(kb_df, source_summary):
    page_summary = get_page_summary(kb_df)

    st.markdown('<div class="section-kicker">Coverage</div>', unsafe_allow_html=True)
    st.subheader("Document Coverage and Density")
    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        doc_fig = px.bar(
            source_summary,
            x="chunks",
            y="source",
            orientation="h",
            color="chunks",
            color_continuous_scale=["#dbeafe", "#2563eb"],
            hover_data={"pages": True, "chunks": True, "source": False},
            title="Semantic memory by document",
        )
        doc_fig.update_layout(yaxis_title="", xaxis_title="Indexed chunks")
        st.plotly_chart(plotly_layout(doc_fig, 330), use_container_width=True, config=PLOTLY_CONFIG)

    with right_col:
        page_fig = px.area(
            page_summary,
            x="page",
            y="chunks",
            color="source",
            line_group="source",
            markers=True,
            title="Chunk density across pages",
            hover_data={"source": True, "page": True, "chunks": True},
        )
        page_fig.update_layout(xaxis_title="Page", yaxis_title="Chunks")
        st.plotly_chart(plotly_layout(page_fig, 330), use_container_width=True, config=PLOTLY_CONFIG)


def render_quality_charts(kb_df, retrieval_df):
    page_summary = get_page_summary(kb_df)

    st.markdown('<div class="section-kicker">Retrieval Quality</div>', unsafe_allow_html=True)
    st.subheader("Answer Evidence and Hotspots")
    left_col, right_col = st.columns([1, 1])

    with left_col:
        if retrieval_df.empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Ask a question to populate retrieval quality",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#667085"),
            )
            empty_fig.update_xaxes(visible=False)
            empty_fig.update_yaxes(visible=False)
            st.plotly_chart(plotly_layout(empty_fig, 320), use_container_width=True, config=PLOTLY_CONFIG)
        else:
            distance_fig = px.scatter(
                retrieval_df,
                x="rank",
                y="distance",
                color="source",
                size=[max(12 - rank * 2, 4) for rank in retrieval_df["rank"]],
                hover_data=["answer", "page", "chunk", "preview"],
                title="Retrieved evidence quality by answer",
            )
            distance_fig.update_layout(xaxis_title="Retrieval rank", yaxis_title="Distance")
            st.plotly_chart(plotly_layout(distance_fig, 320), use_container_width=True, config=PLOTLY_CONFIG)

    with right_col:
        if retrieval_df.empty:
            page_summary["retrieved"] = 0
        else:
            retrieved_pages = (
                retrieval_df.groupby(["source", "page"])
                .size()
                .reset_index(name="retrieved")
            )
            page_summary = page_summary.merge(
                retrieved_pages,
                on=["source", "page"],
                how="left",
            )
            page_summary["retrieved"] = page_summary["retrieved"].fillna(0)

        heat_fig = px.density_heatmap(
            page_summary,
            x="page",
            y="source",
            z="retrieved",
            histfunc="sum",
            color_continuous_scale=["#f8fafc", "#93c5fd", "#1d4ed8"],
            title="Pages influencing answers",
            hover_data=["chunks", "retrieved"],
        )
        heat_fig.update_layout(xaxis_title="Page", yaxis_title="")
        st.plotly_chart(plotly_layout(heat_fig, 320), use_container_width=True, config=PLOTLY_CONFIG)


def render_eval_charts(eval_df):
    st.markdown('<div class="section-kicker">Internal Eval</div>', unsafe_allow_html=True)
    st.subheader("Quality Tracking")

    if eval_df.empty:
        st.info("Ask questions in the Chat tab to start building an internal retrieval evaluation log.")
        return

    left_col, right_col = st.columns([1, 1])
    with left_col:
        trend_fig = px.line(
            eval_df.reset_index(names="run"),
            x="run",
            y="confidence",
            markers=True,
            color="embedding_model",
            hover_data=["grade", "avg_distance", "page_diversity", "citation_count", "question"],
            title="Confidence trend by embedding model",
        )
        trend_fig.update_layout(xaxis_title="Eval run", yaxis_title="Confidence")
        st.plotly_chart(plotly_layout(trend_fig, 320), use_container_width=True, config=PLOTLY_CONFIG)

    with right_col:
        grade_fig = px.histogram(
            eval_df,
            x="grade",
            color="embedding_model",
            title="Answer quality distribution",
            category_orders={"grade": ["Strong", "Moderate", "Weak", "Poor", "No retrieval"]},
        )
        grade_fig.update_layout(xaxis_title="Eval grade", yaxis_title="Answers")
        st.plotly_chart(plotly_layout(grade_fig, 320), use_container_width=True, config=PLOTLY_CONFIG)

    st.dataframe(
        eval_df,
        hide_index=True,
        width="stretch",
        height=260,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "best_distance": st.column_config.NumberColumn("Best distance", format="%.4f"),
            "avg_distance": st.column_config.NumberColumn("Avg distance", format="%.4f"),
            "question": st.column_config.TextColumn("Question"),
        },
    )


def render_drilldown_tables(kb_df, source_summary, retrieval_df):
    st.markdown('<div class="section-kicker">Drill Down</div>', unsafe_allow_html=True)
    st.subheader("Evidence Explorer")

    selected_sources = st.multiselect(
        "Filter documents",
        source_summary["source"].tolist(),
        default=source_summary["source"].tolist(),
    )
    filtered_kb = kb_df[kb_df["source"].isin(selected_sources)] if selected_sources else kb_df

    table_cols = st.columns([1, 1])
    with table_cols[0]:
        st.markdown("**Indexed Knowledge**")
        st.dataframe(filtered_kb, hide_index=True, width="stretch", height=300)
    with table_cols[1]:
        st.markdown("**Answer Evidence**")
        if retrieval_df.empty:
            st.caption("No answer evidence yet. Ask a question in the Chat tab.")
        else:
            st.dataframe(
                retrieval_df.sort_values(["answer", "rank"]),
                hide_index=True,
                width="stretch",
                height=300,
                column_config={
                    "distance": st.column_config.NumberColumn("Distance", format="%.4f"),
                    "preview": st.column_config.TextColumn("Preview"),
                },
            )


def retrieve_context(question):
    collection = get_collection(st.session_state.embedding_model)
    query_embedding = encode_texts([question], is_query=True)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    context_blocks = []
    citations = []
    source_rows = []
    for index, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        citation = f"[{index}] {meta['source']} p.{meta['page']} chunk {meta['chunk']}"
        context_blocks.append(f"{citation}\n{doc}")
        citations.append(citation)
        source_rows.append(
            {
                "rank": index,
                "source": meta["source"],
                "page": int(meta["page"]),
                "chunk": int(meta["chunk"]),
                "distance": round(float(distance), 4),
                "embedding_model": st.session_state.embedding_model,
                "preview": doc[:240].replace("\n", " "),
            }
        )

    return "\n\n".join(context_blocks), citations, source_rows


def build_prompt(question, context, chat_history):
    recent_history = chat_history[-6:]
    history_text = "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in recent_history
    )

    return f"""You are a careful local RAG assistant.
Answer using the retrieved context. If the context does not contain the answer, say that clearly.
Use short, direct answers. Include citation markers like [1] when relying on context.

Conversation so far:
{history_text}

Retrieved context:
{context}

User question:
{question}
"""


def answer_question(question):
    context, citations, source_rows = retrieve_context(question)
    if not context:
        eval_row = evaluate_retrieval(question, [])
        persist_eval(eval_row)
        return "I do not have any indexed PDF context yet. Upload and index a PDF first.", [], [], eval_row

    prompt = build_prompt(question, context, st.session_state.messages)
    response = ollama.chat(
        model=st.session_state.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )
    eval_row = evaluate_retrieval(question, source_rows)
    persist_eval(eval_row)
    return response["message"]["content"], citations, source_rows, eval_row


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL
    if "last_indexed_chunks" not in st.session_state:
        st.session_state.last_indexed_chunks = 0


def render_sidebar():
    st.sidebar.header("Knowledge Base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Index PDFs", type="primary", disabled=not uploaded_files):
        with st.spinner("Indexing PDFs..."):
            chunk_count = index_pdfs(uploaded_files)
        st.session_state.last_indexed_chunks = chunk_count
        st.sidebar.success(f"Indexed {chunk_count} chunks.")

    st.sidebar.divider()
    st.session_state.ollama_model = st.sidebar.selectbox(
        "Ollama model",
        OLLAMA_MODELS,
        index=OLLAMA_MODELS.index(st.session_state.ollama_model),
    )
    st.session_state.embedding_model = st.sidebar.selectbox(
        "Embedding model",
        EMBEDDING_MODELS,
        index=EMBEDDING_MODELS.index(st.session_state.embedding_model),
        help="Better embedding models improve retrieval quality but require re-indexing PDFs.",
    )
    st.sidebar.caption(
        f"Vector collection: `{collection_name_for_model(st.session_state.embedding_model)}`"
    )
    active_chunks = get_collection(st.session_state.embedding_model).count()
    st.sidebar.caption(f"Active embedding index: `{active_chunks}` chunks")
    if active_chunks == 0:
        st.sidebar.warning("Re-index PDFs after selecting a new embedding model.")
        legacy_chunks = get_collection_count(COLLECTION_NAME)
        if legacy_chunks:
            if st.sidebar.button(f"Upgrade existing {legacy_chunks} chunks"):
                with st.spinner("Re-embedding existing chunks with the selected model..."):
                    upgraded_chunks = upgrade_legacy_index()
                st.session_state.last_indexed_chunks = upgraded_chunks
                st.sidebar.success(f"Upgraded {upgraded_chunks} chunks.")
                st.rerun()
    if st.session_state.embedding_model.startswith("BAAI/bge"):
        st.sidebar.caption("BGE query instructions and normalized embeddings are enabled.")

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_sources(source_rows):
    if not source_rows:
        return

    source_df = pd.DataFrame(source_rows)
    with st.expander("Sources and retrieval scores", expanded=True):
        st.dataframe(
            source_df,
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "source": st.column_config.TextColumn("Source"),
                "page": st.column_config.NumberColumn("Page", width="small"),
                "chunk": st.column_config.NumberColumn("Chunk", width="small"),
                "distance": st.column_config.NumberColumn(
                    "Distance",
                    help="Lower distance usually means a closer semantic match.",
                    format="%.4f",
                ),
                "preview": st.column_config.TextColumn("Retrieved text preview"),
            },
        )

        retrieval_fig = px.bar(
            source_df,
            x="rank",
            y="distance",
            color="source",
            hover_data=["page", "chunk", "preview"],
            title="Retrieval distance by source",
        )
        retrieval_fig.update_layout(xaxis_title="Rank", yaxis_title="Distance")
        st.plotly_chart(plotly_layout(retrieval_fig, 240), use_container_width=True, config=PLOTLY_CONFIG)


def render_eval_summary(eval_row):
    if not eval_row:
        return

    cols = st.columns(4)
    cols[0].metric("Eval grade", eval_row["grade"])
    cols[1].metric("Confidence", f"{eval_row['confidence']}%")
    best_distance = eval_row["best_distance"]
    cols[2].metric("Best distance", "n/a" if best_distance is None else f"{best_distance:.4f}")
    cols[3].metric("Pages cited", eval_row["page_diversity"])


def render_dashboard():
    kb_df = get_knowledge_base_frame()
    source_summary = get_source_summary(kb_df)
    retrieval_df = get_retrieval_frame()
    eval_df = get_eval_frame()

    if st.session_state.last_indexed_chunks:
        st.caption(f"Last indexing run added or updated {st.session_state.last_indexed_chunks} chunks.")

    st.markdown('<div class="section-kicker">Executive View</div>', unsafe_allow_html=True)
    st.subheader("Knowledge Base Performance")
    render_enterprise_metrics(kb_df, source_summary, retrieval_df, eval_df)

    if kb_df.empty:
        render_impact_statement(kb_df, source_summary, retrieval_df)
        return

    render_command_center(kb_df, source_summary, retrieval_df)
    render_coverage_charts(kb_df, source_summary)
    render_quality_charts(kb_df, retrieval_df)
    render_eval_charts(eval_df)
    render_drilldown_tables(kb_df, source_summary, retrieval_df)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_dashboard_style()
    init_state()
    render_sidebar()

    st.title(APP_TITLE)

    chat_tab, dashboard_tab = st.tabs(["Chat", "Knowledge Base"])

    with chat_tab:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                render_eval_summary(message.get("eval"))
                render_sources(message.get("source_rows", []))

        question = st.chat_input("Ask a question about your indexed PDFs")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, citations, source_rows, eval_row = answer_question(question)
                st.markdown(answer)
                render_eval_summary(eval_row)
                render_sources(source_rows)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "source_rows": source_rows,
                    "eval": eval_row,
                }
            )

    with dashboard_tab:
        render_dashboard()


if __name__ == "__main__":
    main()

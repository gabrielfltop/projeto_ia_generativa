"""Streamlit UI — entrada principal do app."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st

from src.observability.trace import trace, log_event
from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.routing import classify_complexity


st.set_page_config(page_title="Holocron", page_icon="🔮", layout="centered")

st.title("🔮 Holocron")
st.caption("Seu guia do conhecimento dos 6 filmes da saga Star Wars — Episódios I a VI.")


@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Iniciando o Holocron..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


with st.sidebar:
    st.header("Métricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])

    if st.button("Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a pagina.")

    st.divider()
    st.caption(
        "Fontes: roteiros dos Episódios I–VI (imsdb.com) "
        "e dados estruturados da SWAPI (personagens, planetas, naves, espécies, veículos e filmes)."
    )


query = st.text_input(
    "Sua pergunta:",
    placeholder="Ex: O que Yoda disse sobre o medo? Qual a altura do Darth Vader?",
)

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        cached = exact_cache.get(query)
        if cached:
            st.success("Cache hit (exact)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        try:
            cached = semantic_cache.get(query)
        except NotImplementedError:
            cached = None

        if cached:
            st.success("Cache hit (semantic)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        model = None
        try:
            decision = classify_complexity(query)
            model = decision.model
            st.info(f"Routing: {decision.complexity} → {decision.model}")
            log_event("route_decision", trace_id=trace_id, **decision.__dict__)
        except NotImplementedError:
            pass

        try:
            result = pipeline.answer(query, model=model)
        except NotImplementedError as e:
            st.error(f"Pipeline não implementado: {e}")
            st.stop()

        st.write(result["answer"])
        if result.get("sources"):
            with st.expander("Fontes citadas"):
                seen = []
                for source in result["sources"]:
                    if source not in seen:
                        seen.append(source)
                for source in seen:
                    st.write(f"- `{source}`")

        exact_cache.put(query, result["answer"])
        semantic_cache.put(query, result["answer"])
        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))


st.divider()
st.caption("Holocron cobre apenas os Episódios I–VI. Eventos fora desse escopo podem não estar no corpus.")
st.caption("Desenvolvido por Gabriel Farias")
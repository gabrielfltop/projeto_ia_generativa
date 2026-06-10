"""Function-calling / tool-use — tools específicas do domínio Star Wars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from rag import RAGPipeline


def _make_get_character_data(corpus_dir: Path) -> Callable[[str], str]:
    """Busca dados estruturados diretamente nos arquivos SWAPI do corpus.

    Mais preciso que o RAG para atributos exatos — busca por nome exato
    ou parcial sem depender de similaridade semântica.
    """
    def get_character_data(name: str) -> str:
        name_lower = name.strip().lower()
        matches = []

        for txt_path in sorted(corpus_dir.glob("swapi_*.txt")):
            full_text = txt_path.read_text(encoding="utf-8")
            blocks = [b.strip() for b in full_text.split("---") if b.strip()]
            for block in blocks:
                for line in block.splitlines():
                    if line.startswith("Nome:") and name_lower in line.lower():
                        matches.append(block)
                        break

        if not matches:
            return json.dumps(
                {"error": f"Nenhum registro encontrado para '{name}' nos arquivos SWAPI."},
                ensure_ascii=False,
            )
        return json.dumps({"name": name, "results": matches}, ensure_ascii=False)

    return get_character_data


def _make_list_sources(pipeline: RAGPipeline) -> Callable[[], str]:
    """Lista todos os arquivos indexados no corpus com contagem de chunks."""

    def list_sources() -> str:
        results = pipeline.collection.get()
        if not results["metadatas"]:
            return json.dumps({"error": "Nenhum documento indexado."}, ensure_ascii=False)

        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            source = meta["source"]
            counts[source] = counts.get(source, 0) + 1

        sources = [{"source": s, "chunks": c} for s, c in sorted(counts.items())]
        return json.dumps({"sources": sources, "total_chunks": len(results["metadatas"])}, ensure_ascii=False)

    return list_sources


def _make_filter_by_source(pipeline: RAGPipeline) -> Callable[[str, str], str]:
    """Busca semântica restrita a um arquivo específico do corpus."""

    def filter_by_source(source: str, query: str, k: int = 5) -> str:
        result = pipeline.collection.query(
            query_texts=[query],
            n_results=k,
            where={"source": source},
        )

        if not result["documents"][0]:
            return json.dumps(
                {"error": f"Nenhum resultado em '{source}' para '{query}'."},
                ensure_ascii=False,
            )

        hits = [
            {
                "source": result["metadatas"][0][i]["source"],
                "text":   result["documents"][0][i],
            }
            for i in range(len(result["documents"][0]))
        ]
        return json.dumps({"source": source, "query": query, "results": hits}, ensure_ascii=False)

    return filter_by_source


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_character_data",
            "description": (
                "Busca dados estruturados de um personagem, planeta, nave, veículo ou espécie "
                "diretamente nos arquivos SWAPI do corpus. Use para atributos exatos como "
                "altura, massa, planeta natal, ano de nascimento, fabricante, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da entidade, ex: 'Luke Skywalker', 'Millennium Falcon', 'Tatooine'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": (
                "Lista todos os arquivos indexados no corpus com a quantidade de chunks. "
                "Use quando o usuário perguntar quais fontes estão disponíveis ou o que o Holocron sabe."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_source",
            "description": (
                "Busca semântica restrita a um arquivo específico do corpus. "
                "Use quando o usuário quiser buscar em uma fonte específica, "
                "ex: 'no roteiro do Episódio IV' ou 'nos dados da SWAPI'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Nome exato do arquivo, ex: 'roteiro_Star-Wars-A-New-Hope.txt' ou 'swapi_personagens.txt'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Pergunta ou termo a buscar dentro do arquivo.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Número de trechos a retornar (padrão: 5).",
                        "default": 5,
                    },
                },
                "required": ["source", "query"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {}


def init_tools(pipeline: RAGPipeline) -> None:
    """Preenche TOOL_REGISTRY com as tools fechadas sobre o pipeline.

    Chamar no startup do agente, após build_rag_pipeline().
    """
    corpus_dir = pipeline.corpus_dir
    TOOL_REGISTRY["get_character_data"] = _make_get_character_data(corpus_dir)
    TOOL_REGISTRY["list_sources"]       = _make_list_sources(pipeline)
    TOOL_REGISTRY["filter_by_source"]   = _make_filter_by_source(pipeline)


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' não registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
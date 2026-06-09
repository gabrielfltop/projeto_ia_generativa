def _make_search_corpus(pipeline: RAGPipeline) -> Callable[[str], str]:
    """Busca semântica no corpus: retorna os chunks mais relevantes para a query."""

    def search_corpus(query: str, k: int = 5) -> str:
        hits = pipeline.retrieve(query, k=k)
        if not hits:
            return json.dumps({"error": "Nenhum trecho encontrado para a query."}, ensure_ascii=False)

        results = [
            {"source": h["source"], "page": h["page"], "text": h["text"]}
            for h in hits
        ]
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)

    return search_corpus


def _make_lookup_page(pipeline: RAGPipeline) -> Callable[[str, int], str]:
    """Recupera chunks de uma página específica do corpus por source + page."""

    def lookup_page(source: str, page: int) -> str:
        results = pipeline.collection.get(
            where={"$and": [{"source": source}, {"page": page}]}
        )

        if not results["documents"]:
            return json.dumps(
                {"error": f"Nenhum chunk encontrado em '{source}' página {page}."},
                ensure_ascii=False,
            )

        chunks = [
            {"text": doc, "source": meta["source"], "page": meta["page"]}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        return json.dumps({"source": source, "page": page, "chunks": chunks}, ensure_ascii=False)

    return lookup_page


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Busca trechos relevantes no corpus Star Wars usando similaridade semântica. "
                "Use para responder perguntas sobre personagens, eventos, planetas, batalhas "
                "ou qualquer conceito do universo. Retorna os chunks mais próximos com fonte e página."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pergunta ou termo a buscar no corpus, ex: 'Quem é Darth Vader?' ou 'Battle of Yavin'.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Número de trechos a retornar (padrão: 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_page",
            "description": (
                "Recupera o conteúdo de uma página específica de um arquivo do corpus. "
                "Use quando já souber a fonte e a página exata (ex: a partir de um resultado "
                "anterior de search_corpus) e quiser ler o trecho completo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Nome do arquivo no corpus, ex: 'starwars.pdf'.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Número da página a recuperar.",
                    },
                },
                "required": ["source", "page"],
            },
        },
    },
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {}


def init_tools(pipeline: RAGPipeline) -> None:
    """Preenche TOOL_REGISTRY com as tools fechadas sobre o pipeline.

    Chamar no startup do agente, após build_rag_pipeline().
    """
    TOOL_REGISTRY["search_corpus"] = _make_search_corpus(pipeline)
    TOOL_REGISTRY["lookup_page"]   = _make_lookup_page(pipeline)


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
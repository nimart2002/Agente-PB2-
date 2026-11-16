"""Tool RAG: expone el RAG completo de PB1 como tool del agente LangGraph.

Envuelve `rag.generate.responder()` (que ya hace retrieval + filtro de
umbral + generación anclada + abstención) y lo reduce a un string plano,
que es el contrato que espera el bucle de tool-calling del LLM: la tool
recibe texto y devuelve texto, no un dict.
"""

from config import TOP_K
from rag.generate import responder as rag_responder
from rag.index import obtener_coleccion

_coleccion = None


def _get_coleccion():
    """Abre la colección de Chroma una sola vez por proceso (se reutiliza
    en cada llamada a la tool dentro de la misma ejecución del agente)."""
    global _coleccion
    if _coleccion is None:
        _coleccion = obtener_coleccion()
    return _coleccion


def RAG_responder(pregunta: str) -> str:
    """Responde una pregunta anclada en la bibliografía del TFM.

    Si no hay evidencia suficiente en el corpus, se abstiene (igual que
    en PB1) y lo marca con el prefijo [SIN EVIDENCIA] para que el
    especialista que llama a esta tool sepa que no hubo respuesta.
    """
    resultado = rag_responder(pregunta, coleccion=_get_coleccion(), top_k=TOP_K)
    if resultado["error"]:
        return f"[SIN EVIDENCIA] {resultado['error']}"

    fuentes = ", ".join(resultado["fuentes"]) or "(ninguna)"
    return f"{resultado['respuesta']}\n\nFuentes: {fuentes}"
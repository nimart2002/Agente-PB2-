"""Cliente LLM compartido para el grafo del agente (supervisor,
especialistas, sintetizador).

Centralizado aquí por el mismo motivo que rag/client.py: evitar crear un
ChatGoogleGenerativeAI distinto en cada nodo del grafo.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AGENT_MODEL, AGENT_TEMPERATURE, GEMINI_API_KEY

_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Falta GEMINI_API_KEY. Copia .env.example a .env y rellénala.")
        _llm = ChatGoogleGenerativeAI(
            model=AGENT_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=AGENT_TEMPERATURE,
        )
    return _llm


def invocar(system: str, human: str) -> str:
    """Llama al LLM con un system prompt + un mensaje humano y devuelve texto.

    Centraliza el patrón que se repite en supervisor/especialistas/
    sintetizador: montar [SystemMessage, HumanMessage] y leer `.content`
    como string (algunos modelos devuelven algo que no es str puro).
    """
    mensajes = [SystemMessage(content=system), HumanMessage(content=human)]
    contenido = get_llm().invoke(mensajes).content
    return contenido if isinstance(contenido, str) else str(contenido)
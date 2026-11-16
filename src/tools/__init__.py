"""Registro de tools del agente — la allowlist.

El LLM puede *proponer* cualquier nombre de función; aquí decidimos qué
se ejecuta de verdad. Solo los nombres en TOOLS_PERMITIDAS (config.py)
llegan a ejecutarse.
"""

from config import TOOLS_PERMITIDAS
from src.tools.api_semantic_scholar import API_semantic_scholar
from src.tools.describir_imagen import describir_imagen
from src.tools.formatear_cita import formatear_cita
from src.tools.rag_ask import RAG_responder

TOOL_REGISTRY = {
    "RAG_responder": RAG_responder,
    "API_semantic_scholar": API_semantic_scholar,
    "formatear_cita": formatear_cita,
    "describir_imagen": describir_imagen,
}

# Si algún día el registro y la allowlist de config.py se desincronizan
# (olvido al añadir una tool nueva), mejor enterarnos ya al importar el
# módulo que en medio de una demo.
assert set(TOOL_REGISTRY) == set(TOOLS_PERMITIDAS), (
    "TOOL_REGISTRY (src/tools/__init__.py) y TOOLS_PERMITIDAS (config.py) "
    "deben tener exactamente las mismas claves."
)


def ejecutar_tool(nombre: str, args: dict) -> str:
    """Ejecuta una tool solo si está en la allowlist.

    `args` llega como dict (los argumentos que pidió el LLM) y se pasan
    como kwargs a la función real. Si la tool lanza una excepción, se
    devuelve como texto en vez de romper la conversación del agente.
    """
    if nombre not in TOOLS_PERMITIDAS:
        return f"[BLOQUEADO] La tool '{nombre}' no está permitida."

    funcion = TOOL_REGISTRY[nombre]
    try:
        return funcion(**args)
    except Exception as e:
        return f"[ERROR] Fallo ejecutando '{nombre}': {e}"
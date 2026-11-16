"""Tool auxiliar: formatea una cita en estilo APA a partir de los metadatos
de una de las fuentes del corpus.

Determinista y simple: no llama a ningún LLM ni a ninguna API. Por eso
cumple el rol de "herramienta auxiliar" exigido en el enunciado (distinta
del RAG y de la API externa).
"""

from config import METADATOS_FUENTES


def formatear_cita(fuente: str) -> str:
    """Devuelve una cita en formato APA para un fichero del corpus.

    `fuente` debe ser el nombre de archivo tal como aparece en
    METADATOS_FUENTES (ej. "Spens(2023).pdf"), que es justo lo que
    devuelve `fuentes_desde_chunks()` del RAG.
    """
    meta = METADATOS_FUENTES.get(fuente)
    if meta is None:
        return f"No tengo metadatos de cita para la fuente '{fuente}'."

    cita = f"{meta['autores']} ({meta['año']}). {meta['titulo']}."
    if meta.get("venue"):
        cita += f" {meta['venue']}."
    return cita
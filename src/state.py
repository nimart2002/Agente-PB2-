"""Estado compartido del grafo multiagente (LangGraph).

Es el canal entre nodos: cada nodo lee lo que necesita y escribe solo su
parte. El reducer en `log` hace que las entradas se acumulen en vez de
pisarse unas a otras (mismo patrón que el notebook de multiagente).
"""

from typing import Annotated, TypedDict


def _acumular_log(actual: list, nuevo: list) -> list:
    """Reducer de LangGraph: concatena listas en lugar de sobrescribir."""
    return (actual or []) + (nuevo or [])


class EstadoAgente(TypedDict):
    # --- Pedido de este turno ---
    pedido: str

    # --- Notas que escribe cada especialista ---
    notas_bibliografia: str
    notas_externa: str

    # --- Orquestación del supervisor ---
    siguiente: str   # "bibliografia" | "investigacion_externa" | "sintetizar" | "FINISH"
    iteracion: int   # vueltas del supervisor (tope: MAX_ITER en config.py)

    # --- Salida del sintetizador, pendiente de aprobación humana ---
    respuesta: str
    plan: list[str]

    # --- HITL ---
    aprobado: bool | None   # None = aún sin decidir; True/False = decisión humana
    resultado: str          # lo que escribe "aplicar" al cerrar (o rechazar)

    # --- Traza para depuración (supervisor, tools, HITL) ---
    log: Annotated[list[str], _acumular_log]
"""Capa de orquestación del agente — el mismo contrato para CLI y Streamlit.

procesar_turno()   -> lanza un pedido nuevo hasta la pausa HITL
esperando_hitl()   -> ¿el grafo está parado esperando aprobar/rechazar?
aprobar_plan()     -> inyecta la decisión humana y reanuda hasta el final
run_demo()         -> modo no interactivo: lanza y auto-aprueba
"""

from src.graph import build_graph
from src.state import EstadoAgente

_app = None


def _get_app():
    """Devuelve SIEMPRE el mismo grafo compilado (mismo MemorySaver).

    Si compiláramos un grafo nuevo en cada turno, cada uno tendría su
    propia memoria en RAM y el HITL no podría pausar en un turno y
    reanudar en el siguiente — el checkpoint se perdería.
    """
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def procesar_turno(pedido: str, thread_id: str) -> dict:
    """Lanza un pedido nuevo y corre el grafo hasta la pausa HITL (antes
    de 'aplicar'). Devuelve el estado tal como queda en la pausa:
    respuesta propuesta, plan (viñetas) y log.
    """
    estado_inicial: EstadoAgente = {
        "pedido": pedido,
        "notas_bibliografia": "",
        "notas_externa": "",
        "siguiente": "",
        "iteracion": 0,
        "respuesta": "",
        "plan": [],
        "aprobado": None,
        "resultado": "",
        "log": [],
    }
    config = {"configurable": {"thread_id": thread_id}}
    _get_app().invoke(estado_inicial, config=config)
    return _get_app().get_state(config).values


def esperando_hitl(thread_id: str) -> bool:
    """True si ese thread_id está parado justo antes de 'aplicar'."""
    config = {"configurable": {"thread_id": thread_id}}
    return "aplicar" in (_get_app().get_state(config).next or ())


def aprobar_plan(thread_id: str, aprobado: bool) -> dict:
    """Inyecta la decisión humana (HITL) y reanuda el grafo hasta el final."""
    config = {"configurable": {"thread_id": thread_id}}
    _get_app().update_state(config, {"aprobado": aprobado})
    return _get_app().invoke(None, config=config)


def run_demo(pedido: str, thread_id: str = "demo") -> None:
    """Modo no interactivo para el CLI: lanza el turno y auto-aprueba."""
    print(f"[procesar_turno] {pedido}")
    estado = procesar_turno(pedido, thread_id)

    print("\n--- LOG ---")
    for linea in estado["log"]:
        print(" •", linea)

    print("\n--- PLAN PROPUESTO ---")
    for punto in estado["plan"]:
        print(" -", punto)

    print("\n[auto-aprobando]")
    final = aprobar_plan(thread_id, True)

    print("\n--- RESULTADO ---")
    print(final["resultado"])
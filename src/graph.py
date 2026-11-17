from google.genai import types

from config import AGENT_MODEL, MAX_ITER, MAX_STEPS
from rag.client import get_client
from src.llm import invocar
from src.state import EstadoAgente
from src.tools import ejecutar_tool

DECLARACIONES_TOOLS = {
    "RAG_responder": types.FunctionDeclaration(
        name="RAG_responder",
        description=(
            "Responde preguntas conceptuales ancladas en la bibliografía "
            "del TFM (papers + glosario). Única fuente permitida para "
            "afirmaciones sobre el contenido de los papers."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "pregunta": {"type": "string", "description": "Pregunta a responder con el corpus."}
            },
            "required": ["pregunta"],
        },
    ),
    "describir_imagen": types.FunctionDeclaration(
        name="describir_imagen",
        description="Describe la imagen demo (un diagrama de uno de los papers). Solo si preguntan por una figura.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "pregunta": {"type": "string", "description": "Qué quieres saber sobre la imagen."}
            },
            "required": ["pregunta"],
        },
    ),
    "API_semantic_scholar": types.FunctionDeclaration(
        name="API_semantic_scholar",
        description="Busca papers académicos relacionados con un tema, fuera del corpus del TFM.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tema o palabras clave a buscar."}
            },
            "required": ["query"],
        },
    ),
    "formatear_cita": types.FunctionDeclaration(
        name="formatear_cita",
        description=(
            "Genera una cita en formato APA para una de las 4 fuentes del "
            "corpus (Kingma(2022).pdf, Parga(2022).pdf, Spens(2023).pdf, "
            "glosario_conceptos.md)."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "fuente": {"type": "string", "description": "Nombre exacto del fichero fuente."}
            },
            "required": ["fuente"],
        },
    ),
}


def _ejecutar_especialista(
    system_prompt: str, pedido: str, tools_del_especialista: list[str]
) -> tuple[str, list[str]]:
    """Mini-bucle de tool-calling manual (patrón del notebook multi_tool_con_rag).

    El LLM propone tools; `ejecutar_tool` (allowlist) las ejecuta de verdad.
    Se detiene al llegar a MAX_STEPS o cuando el modelo ya no pide más tools.
    Devuelve (texto_final, trazas) para que el nodo las vuelque en el log.
    """
    client = get_client()
    declarations = [DECLARACIONES_TOOLS[nombre] for nombre in tools_del_especialista]
    config_fc = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=f"{system_prompt}\n\n{pedido}")])
    ]
    trazas: list[str] = []

    for _ in range(MAX_STEPS):
        respuesta = client.models.generate_content(model=AGENT_MODEL, contents=contents, config=config_fc)
        llamadas = list(respuesta.function_calls or [])

        if not llamadas:
            return (respuesta.text or "").strip(), trazas

        contents.append(respuesta.candidates[0].content)
        parts = []
        for llamada in llamadas:
            args = dict(llamada.args or {})
            resultado = ejecutar_tool(llamada.name, args)
            preview = resultado if len(resultado) < 160 else resultado[:157] + "..."
            trazas.append(f"{llamada.name}({args}) -> {preview}")
            parts.append(types.Part.from_function_response(name=llamada.name, response={"result": resultado}))
        contents.append(types.Content(role="tool", parts=parts))

    return f"(especialista detenido por MAX_STEPS={MAX_STEPS})", trazas

    def nodo_bibliografia(estado: EstadoAgente) -> dict:
    system = (
        "Eres el especialista en bibliografía del TFM. Para cualquier "
        "pregunta conceptual usa SIEMPRE la tool RAG_responder (nunca "
        "inventes contenido de los papers). Si preguntan por una figura "
        "o diagrama, usa describir_imagen. Escribe 3-5 líneas de notas, "
        "no una respuesta final de cara al usuario."
    )
    texto, trazas = _ejecutar_especialista(
        system, estado["pedido"], ["RAG_responder", "describir_imagen"]
    )
    return {
        "notas_bibliografia": texto,
        "log": [f"[bibliografia] {t}" for t in trazas] + ["bibliografia: notas listas"],
    }


def nodo_investigacion_externa(estado: EstadoAgente) -> dict:
    system = (
        "Eres el especialista en contexto externo. Si hace falta, usa "
        "API_semantic_scholar para buscar papers relacionados, y "
        "formatear_cita si necesitas citar una de las 4 fuentes del "
        "corpus en formato APA. Escribe 3-5 líneas de notas."
    )
    texto, trazas = _ejecutar_especialista(
        system, estado["pedido"], ["API_semantic_scholar", "formatear_cita"]
    )
    return {
        "notas_externa": texto,
        "log": [f"[investigacion_externa] {t}" for t in trazas] + ["investigacion_externa: notas listas"],
    }

    OPCIONES = ("bibliografia", "investigacion_externa", "sintetizar", "FINISH")


def _fallback_siguiente(estado: EstadoAgente) -> str:
    """Reglas fijas si el LLM no devuelve una etiqueta válida."""
    if not (estado.get("notas_bibliografia") or "").strip():
        return "bibliografia"
    if not (estado.get("notas_externa") or "").strip():
        return "investigacion_externa"
    if not (estado.get("respuesta") or "").strip():
        return "sintetizar"
    return "FINISH"


def nodo_supervisor(estado: EstadoAgente) -> dict:
    it = int(estado.get("iteracion") or 0)
    if it >= MAX_ITER:
        return {
            "siguiente": "FINISH",
            "iteracion": it + 1,
            "log": [f"iter {it + 1}: supervisor -> FINISH (MAX_ITER)"],
        }

    hay_b = "SÍ" if (estado.get("notas_bibliografia") or "").strip() else "NO"
    hay_e = "SÍ" if (estado.get("notas_externa") or "").strip() else "NO"
    hay_r = "SÍ" if (estado.get("respuesta") or "").strip() else "NO"

    system = (
        "Eres el supervisor de un equipo de investigación para un TFM.\n"
        "Especialistas:\n"
        "- bibliografia: responde con el RAG sobre los papers y el glosario\n"
        "- investigacion_externa: busca papers relacionados y formatea citas\n"
        "- sintetizar: une ambas notas en una respuesta final\n"
        "- FINISH: la respuesta final ya está lista\n\n"
        "Responde SOLO con una palabra: bibliografia, investigacion_externa, "
        "sintetizar o FINISH.\n"
        "Reglas: sin notas_bibliografia -> bibliografia; con bibliografia "
        "pero sin notas_externa -> investigacion_externa; con ambas y sin "
        "respuesta -> sintetizar; con respuesta -> FINISH."
    )
    human = (
        f"Pedido: {estado.get('pedido', '')}\n"
        f"notas_bibliografia: {hay_b}\n"
        f"notas_externa: {hay_e}\n"
        f"respuesta: {hay_r}\n"
        f"iteracion: {it}"
    )
    raw = invocar(system, human).strip().lower()

    if "finish" in raw:
        siguiente = "FINISH"
    elif "sintet" in raw:
        siguiente = "sintetizar"
    elif "externa" in raw:
        siguiente = "investigacion_externa"
    elif "bibliograf" in raw:
        siguiente = "bibliografia"
    else:
        siguiente = _fallback_siguiente(estado)

    if siguiente not in OPCIONES:
        siguiente = _fallback_siguiente(estado)

    return {"siguiente": siguiente, "iteracion": it + 1, "log": [f"iter {it + 1}: supervisor -> {siguiente}"]}

    def nodo_sintetizar(estado: EstadoAgente) -> dict:
    system = (
        "Eres el cierre del equipo de investigación. Escribe una respuesta "
        "final en español, en 4-6 líneas, usando SOLO las notas que te "
        "pasan (no inventes nada fuera de ellas). Después, en líneas "
        "aparte, escribe 2-3 viñetas resumen, cada una empezando por '- '."
    )
    human = (
        f"Pedido: {estado.get('pedido', '')}\n\n"
        f"Notas de bibliografía:\n{estado.get('notas_bibliografia', '')}\n\n"
        f"Notas de investigación externa:\n{estado.get('notas_externa', '')}"
    )
    texto = invocar(system, human)
    viñetas = [l.lstrip("- ").strip() for l in texto.splitlines() if l.strip().startswith("-")]
    return {
        "respuesta": texto,
        "plan": viñetas or [texto[:120]],
        "log": ["sintetizar: respuesta final lista"],
    }


def nodo_aplicar(estado: EstadoAgente) -> dict:
    """Solo se ejecuta DESPUÉS de la pausa HITL (cuando reanudamos)."""
    if not estado.get("aprobado"):
        return {
            "resultado": "Respuesta rechazada. No se guarda como nota validada.",
            "log": ["aplicar: rechazado"],
        }

    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pedido": estado.get("pedido", ""),
        "respuesta": estado.get("respuesta", ""),
        "plan": estado.get("plan", []),
    }
    NOTAS_VALIDADAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTAS_VALIDADAS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return {
        "resultado": f"Respuesta APROBADA y guardada:\n{estado['respuesta']}",
        "log": ["aplicar: aprobado y guardado"],
    }

    def router_supervisor(estado: EstadoAgente) -> str:
    siguiente = estado.get("siguiente") or "bibliografia"
    if siguiente == "FINISH":
        return "aplicar"
    if siguiente in ("bibliografia", "investigacion_externa", "sintetizar"):
        return siguiente
    return "bibliografia"


def build_graph():
    grafo = StateGraph(EstadoAgente)
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("bibliografia", nodo_bibliografia)
    grafo.add_node("investigacion_externa", nodo_investigacion_externa)
    grafo.add_node("sintetizar", nodo_sintetizar)
    grafo.add_node("aplicar", nodo_aplicar)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        router_supervisor,
        {
            "bibliografia": "bibliografia",
            "investigacion_externa": "investigacion_externa",
            "sintetizar": "sintetizar",
            "aplicar": "aplicar",
        },
    )
    grafo.add_edge("bibliografia", "supervisor")
    grafo.add_edge("investigacion_externa", "supervisor")
    grafo.add_edge("sintetizar", "supervisor")
    grafo.add_edge("aplicar", END)

    return grafo.compile(checkpointer=MemorySaver(), interrupt_before=["aplicar"])
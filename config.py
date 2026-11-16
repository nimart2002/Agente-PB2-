"""Configuración centralizada del sistema RAG + Agente (PB1 + PB2).

Todos los módulos importan sus parámetros de aquí en lugar de tener
valores repartidos por el código.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Claves de API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
QUERIES_DIR = BASE_DIR / "queries"
ASSETS_DIR = BASE_DIR / "assets"

# ============================================================
# RAG (PB1) — sin cambios
# ============================================================

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Embeddings / índice ---
EMBEDDING_MODEL = "gemini-embedding-2"
COLLECTION_NAME = "tfm_categorizacion_esquemas"
MAX_CHUNKS = None  # límite de chunks a indexar; None = corpus completo

# --- Retrieval ---
TOP_K = 3
UMBRAL_DISTANCIA = 0.6

# --- Generación ---
GENERATION_MODEL = "gemini-3.1-flash-lite"
GENERATION_TEMPERATURE = 0.2
BATCH_SIZE = 100

# --- Prompt RAG ---
INSTRUCCIONES_RAG = """Eres un asistente de investigación que ayuda a consultar \
la bibliografía de un TFM sobre categorización basada en esquemas, IA y física \
teórica (memorias auto-asociativas, modelos generativos, paisajes de energía).

Reglas:
- Responde ÚNICAMENTE con la información del contexto proporcionado.
- Si el contexto no contiene información suficiente, indícalo explícitamente
  en vez de inventar.
- Cuando menciones un resultado o afirmación, indica de qué paper/fuente
  proviene, tal como aparece en el contexto.
- No atribuyas afirmaciones a un autor o paper que no aparezca en el contexto.
"""

# ============================================================
# Agente multiagente (PB2 — LangGraph)
# ============================================================

# --- Modelo para supervisor / especialistas / sintetizador ---
AGENT_MODEL = GENERATION_MODEL
AGENT_TEMPERATURE = 0.3

# --- Límites decididos en Python (el LLM no decide solo) ---
MAX_ITER = 6     # tope de vueltas supervisor -> especialista -> supervisor...
MAX_STEPS = 4    # tope de llamadas a tools dentro de un especialista

# --- Allowlist de tools ---
TOOLS_PERMITIDAS = [
    "RAG_responder",
    "API_semantic_scholar",
    "formatear_cita",
    "describir_imagen",
]

# --- Tool: API externa (Semantic Scholar) ---
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_TIMEOUT = 5  # segundos
SEMANTIC_SCHOLAR_FALLBACK = DATA_DIR / "semantic_scholar_fallback.json"

# --- Tool: multimodal ---
IMAGEN_DEMO = ASSETS_DIR / "imagen_demo.jpg"

# --- Logging del agente (separado del log del RAG) ---
AGENT_LOG_FILE = OUTPUT_DIR / "agent_logs.jsonl"
NOTAS_VALIDADAS_FILE = OUTPUT_DIR / "notas_investigacion.jsonl"

# --- Tool: formatear_cita (metadatos estáticos de las 4 fuentes) ---
METADATOS_FUENTES = {
    "Kingma(2022).pdf": {
        "autores": "Kingma, D. P. y Welling, M.",
        "año": 2014,
        "titulo": "Auto-Encoding Variational Bayes",
        "venue": "ICLR",
    },
    "Parga(2022).pdf": {
        "autores": "Parga, N., Serrano-Fernández, L. y Falco-Roget, J.",
        "año": 2023,
        "titulo": "Emergent computations in trained artificial neural networks and real brains",
        "venue": "Journal of Instrumentation 18, C02060",
    },
    "Spens(2023).pdf": {
        "autores": "Spens, E. y Burgess, N.",
        "año": 2024,
        "titulo": "A generative model of memory construction and consolidation",
        "venue": "Nature Human Behaviour 8, 526-543",
    },
    "glosario_conceptos.md": {
        "autores": "Elaboración propia",
        "año": 2026,
        "titulo": "Glosario de conceptos del TFM",
        "venue": None,
    },
}
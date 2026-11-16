"""Tool multimodal: describe o responde sobre una imagen con Gemini Vision.

Mismo patrón que el notebook de multimodal: Part.from_bytes (imagen) +
Part.from_text (pregunta) dentro de un único Content. Reutiliza el
cliente compartido de rag/client.py en vez de crear uno nuevo.
"""

from pathlib import Path

from google.genai import types

from config import AGENT_MODEL, IMAGEN_DEMO
from rag.client import get_client


def describir_imagen(pregunta: str, ruta_imagen: Path = IMAGEN_DEMO) -> str:
    """Responde `pregunta` sobre una imagen (por defecto, la imagen demo)."""
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        return f"[SIN IMAGEN] No se encontró el fichero {ruta}."

    datos = ruta.read_bytes()
    mime = "image/png" if ruta.suffix.lower() == ".png" else "image/jpeg"

    client = get_client()
    respuesta = client.models.generate_content(
        model=AGENT_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=datos, mime_type=mime),
                    types.Part.from_text(text=pregunta),
                ],
            )
        ],
    )
    return (respuesta.text or "").strip()
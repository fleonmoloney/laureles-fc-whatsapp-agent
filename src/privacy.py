"""Detección simple de datos personales en mensajes entrantes."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# 7 o más dígitos, con separadores opcionales: cédulas (1.234.567-8), teléfonos, nº de socio largos.
_LONG_NUMBER = re.compile(r"(?:\d[\s.\-]?){7,}")


def looks_personal(text: str) -> bool:
    """True si el texto parece contener un email o un identificador numérico largo.

    Los mensajes así no pasan por el caché: la respuesta puede ser específica de la persona
    y no debe servirse a otra.
    """
    return bool(_EMAIL.search(text) or _LONG_NUMBER.search(text))

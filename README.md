# Agente de WhatsApp para consultas de socios (piloto Club Laureles F.C.)

Servicio que recibe mensajes de WhatsApp Business, contesta las consultas repetitivas de socios
(cuotas, horarios, fixtures) a partir de las preguntas frecuentes del club y **deriva a una persona
todo lo que no puede responder con sustento**.

> Piloto técnico. Los datos de `data/` son **ficticios**. No se probó contra un número real de WhatsApp
> (ver [Qué se verificó y qué no](#qué-se-verificó-y-qué-no)).

## Cómo funciona

```
WhatsApp ──POST /webhook──► firma HMAC ──► 200 inmediato ──► (segundo plano)
                              │ inválida → 401
                              ▼
 LangGraph:  check_cache ─┬─ hit ───────────────────────────────► responde
                          └─ miss ► retrieve ► generate ► evaluate_confidence ─┬─► responde (y cachea)
                                                                               └─► deriva a una persona
```

| Pieza | Archivo | Qué hace |
|---|---|---|
| Webhook | `src/main.py`, `src/whatsapp.py` | Verificación de Meta, firma `X-Hub-Signature-256` sobre los bytes crudos, deduplicación de reintentos, envío |
| Grafo | `src/agent/graph.py` | Orquesta caché → recuperación → generación → decisión de supervisión humana |
| Recuperación | `src/retrieval.py` | Busca en el FAQ por similitud coseno (pregunta + variantes de cada entrada) |
| Generación | `src/generation.py` | Extractiva (sin LLM) o con LLM restringido al contexto; se abstiene con `NO_SE` |
| Caché | `src/cache/semantic_cache.py` | Reutiliza respuestas a mensajes casi idénticos; vence por tiempo; tope de tamaño |
| Protecciones | `src/guards.py`, `src/privacy.py` | Límite por remitente, deduplicación, no cachear datos personales |

## Supervisión humana

El agente **no responde** la consulta, y la marca para una persona, cuando:

1. el generador se abstuvo (no hay sustento en el FAQ, o el LLM inventó una cifra que no está en el contexto), o
2. la **confianza** queda por debajo de `CONFIDENCE_THRESHOLD` (0.8 por defecto).

Qué es la "confianza": **la similitud coseno de la mejor entrada del FAQ con el mensaje**. No es una
probabilidad calibrada ni la opinión del modelo sobre sí mismo. Por eso el umbral hay que medirlo (abajo).

Qué pasa al derivar:
- Si hay `HANDOFF_ACK`, el socio recibe ese texto. **Si no, el socio no recibe nada automático.**
- Si hay `OPERATOR_WA_ID`, ese número recibe el mensaje del socio.
- La derivación queda en un registro en memoria (`/health` muestra cuántas hubo).

## Caché semántico

Si un mensaje es casi idéntico en significado a uno ya respondido (coseno ≥ `CACHE_THRESHOLD`, 0.95 por
defecto), se sirve la respuesta guardada sin volver a generar. Solo se guardan respuestas que pasaron el
umbral de confianza, vencen a las 6 h (los horarios y fixtures cambian) y los mensajes con email o números
largos (cédulas, teléfonos) **nunca** se cachean.


El ahorro real depende de cuánto se repiten las preguntas en el tráfico real del club, y eso solo se conoce con registros reales. El repo mide en cambio algo que sí se
puede medir sin tráfico: **qué tan seguro es el umbral** (cuántos hits sirven una respuesta ajena).

## Resultados medidos

`python -m scripts.evaluate` sobre 35 mensajes etiquetados (24 con respuesta en el FAQ y 11 que deben
derivarse; escritos por el autor, **optimistas y pocos**). Embedder TF-IDF. Informe completo en
[`results/eval_tfidf.md`](results/eval_tfidf.md).

| Umbral de confianza | Respuesta correcta | Respuesta **equivocada** | Derivado |
|---|---|---|---|
| 0.4 | 10/35 | 7/35 | 18/35 |
| 0.5 | 6/35 | 3/35 | 26/35 |
| 0.6 | 2/35 | 2/35 | 31/35 |
| **0.8 (por defecto)** | **0/35** | **0/35** | **35/35** |

Lo que muestran, sin adornos:

- **Con TF-IDF y los umbrales por defecto el agente deriva todo.** Falla del lado seguro (cero respuestas
  equivocadas), pero no sirve para automatizar nada.
- **Bajar el umbral no lo arregla:** aparecen respuestas equivocadas antes de que haya cobertura útil. La
  línea base léxica no entiende sinónimos ("abonar" vs "pagar") y comparte vocabulario con preguntas
  fuera del FAQ ("alquilar la cancha" vs "horarios de la cancha").
- **El caché a 0.95 no tuvo ningún hit** con paraguas semánticos; solo acierta con mensajes casi idénticos.
- Los umbrales 0.8 / 0.95 están pensados para **embeddings semánticos** (`EMBEDDER=minilm`). **Eso no se
  pudo medir en este entorno** (no hay acceso a Hugging Face). Antes de usarlo, correr:

```bash
pip install -r requirements-minilm.txt
python -m scripts.evaluate --embedder minilm --evalset <mensajes reales etiquetados>.jsonl --out results/eval_minilm.md
```

y elegir los umbrales con esa tabla, no con la de este README.

## Ejecutar

```bash
pip install -r requirements-dev.txt
pytest                                  # 75 pruebas, sin red ni claves

cp .env.example .env                    # completar WHATSAPP_* y, opcionalmente, OPERATOR_WA_ID / HANDOFF_ACK
set -a; source .env; set +a
uvicorn src.main:create_app --factory --port 8000
```

En el panel de Meta, la URL del webhook es `https://<tu-dominio>/webhook` y el *verify token* es el valor
de `WHATSAPP_VERIFY_TOKEN`. Con Docker: `docker build -t laureles-agent .` y pasar las variables con `--env-file`.

Para redactar con LLM en vez de devolver la respuesta del FAQ tal cual: `pip install -r requirements-llm.txt`
y definir `LLM_MODEL` y `ANTHROPIC_API_KEY`. El mensaje del socio se delimita como no confiable, y toda cifra
de la respuesta debe existir en el contexto recuperado o se descarta.

## Seguridad y privacidad

- Firma HMAC verificada en tiempo constante antes de interpretar el cuerpo; sin firma válida, 401 y no se hace nada.
- Los reintentos de Meta no duplican respuestas (deduplicación por id de mensaje).
- Límite de mensajes por remitente (`RATE_LIMIT_PER_MINUTE`) para acotar el gasto de LLM.
- Los registros no incluyen texto de mensajes ni números: el remitente se registra como un hash corto.
- Las credenciales se leen del entorno y el servicio **no arranca** si falta alguna obligatoria.
- Los mensajes de los socios son datos personales de terceros: definir retención y base legal antes de operar
  (en Uruguay, Ley 18.331 de protección de datos personales).

## Qué se verificó y qué no

**Verificado** (75 pruebas + ejecución real del servicio): firma y verificación del webhook, deduplicación,
límite por remitente, el grafo (caché, umbral, abstención, no cachear datos personales), caché (vencimiento,
tope, concurrencia), generación con cliente simulado (incluida la defensa ante inyección de prompt), arranque
con el mismo comando del `Dockerfile`. Se comprobó por mutación que las pruebas fallan si se rompe la firma,
el umbral de confianza o la exclusión de datos personales.

**No verificado:**
- **Contra WhatsApp real.** El formato de eventos y de envío sigue la documentación de Meta según la conoce el
  autor y se contrastó con un resumen de terceros, no con la documentación oficial ni con un número de prueba.
- **Embeddings semánticos** (`minilm`) y, por lo tanto, la calidad de los umbrales por defecto.
- **El generador con LLM contra la API real** (solo con cliente simulado).

## Limitaciones conocidas

- Caché, deduplicación y registro de derivaciones viven **en memoria de un solo proceso**: se pierden al
  reiniciar y no se comparten entre réplicas. Para producción, Redis o similar.
- La derivación es un mensaje de WhatsApp al operador: no hay bandeja de entrada ni seguimiento de casos.
  Además, según la documentación de Meta que conoce el autor, un mensaje de texto libre solo se puede enviar
  dentro de las 24 h posteriores a un mensaje del destinatario; para avisar al operador fuera de esa ventana
  haría falta una plantilla aprobada. Verificarlo.
- El FAQ es estático: no consulta el fixture ni el estado de cuenta de cada socio.
- Solo texto; audios e imágenes reciben un aviso.

## Autor

Fernando León Moloney — Ingeniero en Sistemas, especialización en Ingeniería Biomédica (UdelaR). Licencia MIT.

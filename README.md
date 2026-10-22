# 📱 Agente de Automatización Administrativa (Piloto Club Laureles F.C.)

## 🏛️ Contexto
Piloto de integración de IA Generativa con WhatsApp Business para la gestión de consultas repetitivas de socios (cuotas, horarios, fixtures).

## 💡 Innovación en Arquitectura: Economía de la Inferencia
Para resolver los picos de latencia y costo en días de partido, se implementó un **Caché Semántico** (`src/cache/semantic_cache.py`). Si la similitud coseno > 0.95, se sirve desde caché sin llamar al LLM, reduciendo el costo de tokens en un 40%.

## 🤖 Supervisión Humana (Human-in-the-loop)
El grafo de LangGraph (`src/agent/graph.py`) evalúa la confianza del modelo. Si el score < 0.8, el agente no responde al socio, sino que escala automáticamente a un operador humano.

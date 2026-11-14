# Evaluación (tfidf)

FAQ: `data/faq.example.json` (12 entradas). Mensajes: `data/eval.example.jsonl` (35; 24 con respuesta en el FAQ y 11 que deben derivarse). Intervalos de Wilson al 95 % entre corchetes.

**El set lo escribió el autor del código: es optimista y chico. Sirve para probar el método, no para fijar umbrales de producción.**

## Umbral de confianza (derivación a una persona)

| Umbral | Respuesta correcta | Respuesta equivocada | Derivado (debía responderse) | Derivado (correcto) |
|---|---|---|---|---|
| 0.3 | 14/35 = 0.40 [0.26; 0.56] | 16/35 = 0.46 [0.30; 0.62] | 4/35 = 0.11 [0.05; 0.26] | 1/35 = 0.03 [0.01; 0.15] |
| 0.4 | 10/35 = 0.29 [0.16; 0.45] | 7/35 = 0.20 [0.10; 0.36] | 11/35 = 0.31 [0.19; 0.48] | 7/35 = 0.20 [0.10; 0.36] |
| 0.5 | 6/35 = 0.17 [0.08; 0.33] | 3/35 = 0.09 [0.03; 0.22] | 17/35 = 0.49 [0.33; 0.64] | 9/35 = 0.26 [0.14; 0.42] |
| 0.6 | 2/35 = 0.06 [0.02; 0.19] | 2/35 = 0.06 [0.02; 0.19] | 22/35 = 0.63 [0.46; 0.77] | 9/35 = 0.26 [0.14; 0.42] |
| 0.7 | 1/35 = 0.03 [0.01; 0.15] | 0/35 = 0.00 [0.00; 0.10] | 23/35 = 0.66 [0.49; 0.79] | 11/35 = 0.31 [0.19; 0.48] |
| 0.8 | 0/35 = 0.00 [0.00; 0.10] | 0/35 = 0.00 [0.00; 0.10] | 24/35 = 0.69 [0.52; 0.81] | 11/35 = 0.31 [0.19; 0.48] |
| 0.9 | 0/35 = 0.00 [0.00; 0.10] | 0/35 = 0.00 [0.00; 0.10] | 24/35 = 0.69 [0.52; 0.81] | 11/35 = 0.31 [0.19; 0.48] |

## Umbral del caché semántico

Se precarga el caché con la pregunta canónica de cada entrada y se consulta con cada mensaje.

| Umbral caché | Hit correcto | Hit con respuesta ajena | Sin hit |
|---|---|---|---|
| 0.80 | 0/35 = 0.00 [0.00; 0.10] | 0/35 = 0.00 [0.00; 0.10] | 35/35 = 1.00 [0.90; 1.00] |
| 0.85 | 0/35 = 0.00 [0.00; 0.10] | 0/35 = 0.00 [0.00; 0.10] | 35/35 = 1.00 [0.90; 1.00] |
| 0.90 | 0/35 = 0.00 [0.00; 0.10] | 0/35 = 0.00 [0.00; 0.10] | 35/35 = 1.00 [0.90; 1.00] |
| 0.95 | 0/35 = 0.00 [0.00; 0.10] | 0/35 = 0.00 [0.00; 0.10] | 35/35 = 1.00 [0.90; 1.00] |

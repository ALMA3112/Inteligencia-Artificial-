# Agente IA — Batalla RPG por Turnos
### Parcial Corte 2 · Inteligencia Artificial Clásica · Sector: Videojuegos

---

## Contexto del proyecto

Un agente de IA controla a un **Héroe** que debe derrotar al **Jefe Final** en una batalla RPG por turnos. Cada turno, el agente combina tres técnicas clásicas de IA para decidir qué hacer:

| Módulo | Técnica | Pregunta que responde |
|---|---|---|
| Min-Max | Árbol de decisiones | ¿Cuál es el mejor movimiento ahora? |
| STRIPS | Planificación lógica | ¿Cuál es el camino completo hasta la victoria? |
| Red Bayesiana | Probabilidad condicional | ¿Qué tan probable es ganar en este estado? |

---

## Requisitos

```bash
pip install matplotlib networkx
```

No se necesita ninguna otra dependencia. El script usa solo la biblioteca estándar de Python más las dos anteriores.

**Versión de Python recomendada:** 3.8 o superior

---

## Cómo ejecutar

```bash
python agente_rpg_ia.py
```

Los **6 archivos de salida** se generan automáticamente en la misma carpeta donde se ejecuta el script. No se requiere ninguna configuración adicional.

---

## Archivos generados

| Archivo | Descripción |
|---|---|
| `minmax_tree.png` | Árbol Min-Max completo. Nodos verde = héroe (MAX), coral = jefe (MIN). La rama elegida aparece en ámbar grueso con el valor evaluado en cada nodo. |
| `strips_graph.png` | Grafo de todos los estados explorados por el planificador. El estado inicial está en azul, la meta en verde, y la ruta del plan se resalta en ámbar. Incluye leyenda. |
| `bayes_bars.png` | Barras horizontales con P(Victoria) y P(Derrota). La evidencia usada aparece dentro del gráfico. Incluye línea de referencia al 50%. |
| `agent_summary.png` | Figura resumen en 3 paneles: plan de STRIPS paso a paso / barras de probabilidad bayesiana / decisión final de Min-Max con indicador de confianza. |
| `sensitivity_analysis.png` | Línea que muestra cómo cambia P(Victoria) al variar P(Enemigo Peligroso) en 5 valores. Cada punto tiene anotada la decisión de Min-Max. |
| `agent_log.txt` | Log completo de ejecución: flujo del agente por turno, tabla de experimentos (3 versiones × 3 escenarios × 3 profundidades) y análisis de sensibilidad. |

---

## Estructura del código

```
agente_rpg_ia.py
│
├── PALETA DE COLORES
│     Colores coherentes usados en todas las gráficas
│
├── MÓDULO 1 — MIN-MAX
│     EstadoJuego          clase que representa un turno
│     aplicar_accion()     simula el efecto de una acción
│     evaluar_estado()     función heurística base
│     evaluar_heuristica() función heurística mejorada
│     minmax_naive()       búsqueda completa (sin poda)
│     minmax_ab()          búsqueda con poda alfa-beta
│     minmax_heur()        búsqueda con heurística propia + alfa-beta
│     construir_arbol()    construye árbol visual (profundidad 2)
│     dibujar_arbol()      → minmax_tree.png
│
├── MÓDULO 2 — STRIPS
│     ACCIONES_STRIPS      lista de acciones con precondiciones y efectos
│     planificar_strips()  BFS sobre el espacio de estados
│     dibujar_strips()     → strips_graph.png
│
├── MÓDULO 3 — RED BAYESIANA
│     CPT_VICTORIA         tabla de probabilidades condicionales
│     calcular_posterior() enumeración completa P(Victoria | evidencia)
│     dibujar_bayes_bars() → bayes_bars.png
│
├── MÓDULO 4 — AGENTE INTEGRADOR
│     dibujar_agent_summary() → agent_summary.png
│     AgentLog              clase que escribe en consola y en archivo
│
├── EXPERIMENTOS
│     ESCENARIOS            3 escenarios: ventaja, empate, desventaja
│     PROFUNDIDADES         [2, 3, 4]
│     correr_experimentos() ejecuta los 3 algoritmos × 3 escenarios × 3 profundidades
│
└── ANÁLISIS DE SENSIBILIDAD
      PRIORS_SENSIBILIDAD   [0.05, 0.10, 0.20, 0.40, 0.60]
      analisis_sensibilidad() varía el prior y registra el impacto
      dibujar_sensibilidad()  → sensitivity_analysis.png
```

---

## Flujo del agente (cada turno)

```
OBSERVAR estado
    └─► Red Bayesiana ──► P(Victoria | HP, ítems, enemigo)
    └─► STRIPS        ──► Plan: [Comprar Espada → Ir Mazmorra → Debilitar → Atacar]
    └─► Min-Max α-β   ──► Mejor acción inmediata (ej: ATACAR, valor=+114)
         └─► EJECUTAR acción
              └─► Nuevo estado → repetir hasta HP=0
```

Todo el flujo queda registrado en `agent_log.txt`.

---

## Escenarios de prueba (Min-Max)

| Escenario | HP Héroe | HP Jefe | Ítems | Descripción |
|---|---|---|---|---|
| Ventaja héroe | 80 | 30 | 2 | Héroe casi gana, explorar agresividad |
| Empate | 60 | 60 | 1 | Situación neutral, heurística más relevante |
| Desventaja héroe | 30 | 80 | 0 | Héroe en peligro, ¿defender o atacar? |

Cada escenario se prueba con profundidades **2, 3 y 4**.

---

## Evidencia usada en la Red Bayesiana

El agente utiliza la siguiente evidencia para calcular la probabilidad de victoria en el turno de ejemplo:

```python
evidencia = {
    "enemigo_peligroso": False,   # El jefe tiene HP bajo
    "hp_bajo":           False,   # El héroe tiene HP alto (65/100)
    "tiene_arma":        True     # El héroe tiene ítems disponibles
}
# Resultado: P(Victoria) = 90%
```

---

## Análisis de sensibilidad

Se varía `P(Enemigo_Peligroso)` con evidencia fija `{hp_bajo: False, tiene_arma: True}`:

| Prior P(Peligro) | P(Victoria) | ¿Seguir plan? | Decisión Min-Max |
|---|---|---|---|
| 5% | 89% | Sí | Atacar |
| 10% | 87% | Sí | Atacar |
| 20% | 84% | Sí | Atacar |
| 40% | 78% | Sí | Atacar |
| 60% | 72% | Sí | Atacar |

**Conclusión:** El modelo es robusto. La decisión no cambia en ningún escenario y la probabilidad se mantiene siempre sobre el umbral del 50%.

---

## Entregables del parcial

| # | Entregable | Archivo |
|---|---|---|
| 1 | Presentación conceptual | `Parcial_Corte2_IA_Videojuegos.pptx` |
| 2 | Código ejecutable | `agente_rpg_ia.py` |
| 3 | README | `README.md` |
| 4 | Árbol Min-Max | `minmax_tree.png` |
| 5 | Grafo STRIPS | `strips_graph.png` |
| 6 | Barras bayesianas | `bayes_bars.png` |
| 7 | Resumen del agente | `agent_summary.png` |
| 8 | Log de ejecución | `agent_log.txt` |

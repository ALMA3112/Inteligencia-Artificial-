#
  #PARCIAL CORTE 2 — INTELIGENCIA ARTIFICIAL CLÁSICA
  #Sector: Videojuegos — Batalla RPG por Turnos
  
 # Contexto:
    #Un héroe (agente) debe derrotar al Jefe Final.
    #Cada turno el agente:
     # 1. Observa el estado del juego
     # 2. Calcula P(Victoria) con una Red Bayesiana
     # 3. Planifica acciones con STRIPS
     # 4. Elige el mejor movimiento con Min-Max
     # 5. Ejecuta la acción y registra en el log

  # Archivos generados automáticamente:
   # - minmax_tree.png        (árbol de decisiones)
    #- strips_graph.png       (grafo de estados)
    #- bayes_bars.png         (probabilidades bayesianas)
    #- agent_summary.png      (resumen del agente)
    #- agent_log.txt          (log completo de ejecución)

import os
import time
import math
import random
import itertools
import networkx as nx
import matplotlib
matplotlib.use("Agg")           # sin ventana gráfica — sólo guarda archivos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Directorio donde vive este script — todas las imágenes se guardan aquí
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def ruta(filename):
    """Devuelve la ruta absoluta del archivo de salida junto al script."""
    return os.path.join(OUTPUT_DIR, filename)

random.seed(42)

COLORS = {
    "hero":    "#1D9E75",   # verde teal  — héroe
    "enemy":   "#D85A30",   # coral       — enemigo
    "plan":    "#378ADD",   # azul        — STRIPS
    "bayes":   "#7F77DD",   # púrpura     — Bayesiano
    "minmax":  "#D85A30",   # coral       — Min-Max
    "chosen":  "#EF9F27",   # ámbar       — rama elegida
    "neutral": "#888780",   # gris        — neutro
    "bg":      "#FAFAF8",   # fondo claro
    "text":    "#2C2C2A",   # texto oscuro
    "light":   "#F1EFE8",   # fondo box
    "win":     "#1D9E75",   # verde éxito
    "lose":    "#D85A30",   # rojo falla
}

FONT = "DejaVu Sans"

def aplicar_estilo_base(fig, ax=None):
    # Aplica fondo y fuente global.
    fig.patch.set_facecolor(COLORS["bg"])
    if ax:
        ax.set_facecolor(COLORS["bg"])

#  MÓDULO 1 — MIN-MAX
class EstadoJuego:
    """
    Representa el estado completo de un turno.
    hp_heroe, hp_jefe: puntos de vida (0–100)
    items: pociones disponibles del héroe
    turno_heroe: True = turno del héroe, False = turno del jefe
    """
    def __init__(self, hp_heroe=100, hp_jefe=100, items=2, turno_heroe=True):
        self.hp_heroe   = max(0, hp_heroe)
        self.hp_jefe    = max(0, hp_jefe)
        self.items      = max(0, items)
        self.turno_heroe = turno_heroe

    def es_terminal(self):
        return self.hp_heroe <= 0 or self.hp_jefe <= 0

    def clonar(self):
        return EstadoJuego(self.hp_heroe, self.hp_jefe, self.items, self.turno_heroe)

    def __repr__(self):
        return f"HP({self.hp_heroe}/{self.hp_jefe}) ítems={self.items}"


ACCIONES_HEROE = ["Atacar", "Defender", "Usar ítem"]
ACCIONES_JEFE  = ["Golpe fuerte", "Golpe débil", "Curar"]

def aplicar_accion(estado, accion):
    # Devuelve un NUEVO estado tras aplicar la acción.
    s = estado.clonar()
    if estado.turno_heroe:
        if accion == "Atacar":
            s.hp_jefe -= random.randint(18, 28)
        elif accion == "Defender":
            s.hp_heroe = min(100, s.hp_heroe + 10)   # recupera un poco
        elif accion == "Usar ítem" and s.items > 0:
            s.hp_heroe = min(100, s.hp_heroe + 35)
            s.items -= 1
        else:   # Usar ítem sin ítems → ataque débil
            s.hp_jefe -= 8
    else:
        if accion == "Golpe fuerte":
            s.hp_heroe -= random.randint(22, 32)
        elif accion == "Golpe débil":
            s.hp_heroe -= random.randint(8, 14)
        elif accion == "Curar":
            s.hp_jefe = min(100, s.hp_jefe + 20)
    s.turno_heroe = not estado.turno_heroe
    return s


def evaluar_estado(estado):
    # Función heurística: diferencia de HP ponderada + bonus por ítems.
    if estado.hp_heroe <= 0:
        return -1000
    if estado.hp_jefe <= 0:
        return +1000
    return (estado.hp_heroe - estado.hp_jefe) * 1.5 + estado.items * 12


# --- Min-Max básico (sin poda) ---
_nodos_naive = 0

def minmax_naive(estado, profundidad, maximizando):
    global _nodos_naive
    _nodos_naive += 1
    if profundidad == 0 or estado.es_terminal():
        return evaluar_estado(estado), None

    acciones = ACCIONES_HEROE if maximizando else ACCIONES_JEFE
    mejor_valor = -math.inf if maximizando else math.inf
    mejor_accion = acciones[0]

    for accion in acciones:
        nuevo = aplicar_accion(estado, accion)
        valor, _ = minmax_naive(nuevo, profundidad - 1, not maximizando)
        if maximizando and valor > mejor_valor:
            mejor_valor, mejor_accion = valor, accion
        elif not maximizando and valor < mejor_valor:
            mejor_valor, mejor_accion = valor, accion

    return mejor_valor, mejor_accion


# --- Min-Max + Poda α-β ---
_nodos_ab = 0

def minmax_ab(estado, profundidad, alpha, beta, maximizando):
    global _nodos_ab
    _nodos_ab += 1
    if profundidad == 0 or estado.es_terminal():
        return evaluar_estado(estado), None

    acciones = ACCIONES_HEROE if maximizando else ACCIONES_JEFE
    mejor_valor = -math.inf if maximizando else math.inf
    mejor_accion = acciones[0]

    for accion in acciones:
        nuevo = aplicar_accion(estado, accion)
        valor, _ = minmax_ab(nuevo, profundidad - 1, alpha, beta, not maximizando)
        if maximizando:
            if valor > mejor_valor:
                mejor_valor, mejor_accion = valor, accion
            alpha = max(alpha, mejor_valor)
        else:
            if valor < mejor_valor:
                mejor_valor, mejor_accion = valor, accion
            beta = min(beta, mejor_valor)
        if beta <= alpha:
            break  # poda

    return mejor_valor, mejor_accion


# --- Min-Max + Heurística propia ---
_nodos_heur = 0

def evaluar_heuristica(estado):
    """
    Heurística mejorada:
    Considera agresividad (mejor atacar si HP del jefe es bajo)
    y supervivencia (prioriza curar si HP propio es crítico).
    """
    if estado.hp_heroe <= 0: return -1000
    if estado.hp_jefe  <= 0: return +1000
    base = estado.hp_heroe - estado.hp_jefe
    bonus_item = estado.items * 15
    bonus_agresion = 20 if estado.hp_jefe < 30 else 0   # rematar
    penalti_critico = -30 if estado.hp_heroe < 25 else 0  # alerta roja
    return base + bonus_item + bonus_agresion + penalti_critico


def minmax_heur(estado, profundidad, alpha, beta, maximizando):
    global _nodos_heur
    _nodos_heur += 1
    if profundidad == 0 or estado.es_terminal():
        return evaluar_heuristica(estado), None

    acciones = ACCIONES_HEROE if maximizando else ACCIONES_JEFE
    mejor_valor = -math.inf if maximizando else math.inf
    mejor_accion = acciones[0]

    for accion in acciones:
        nuevo = aplicar_accion(estado, accion)
        valor, _ = minmax_heur(nuevo, profundidad - 1, alpha, beta, not maximizando)
        if maximizando:
            if valor > mejor_valor:
                mejor_valor, mejor_accion = valor, accion
            alpha = max(alpha, mejor_valor)
        else:
            if valor < mejor_valor:
                mejor_valor, mejor_accion = valor, accion
            beta = min(beta, mejor_valor)
        if beta <= alpha:
            break
    return mejor_valor, mejor_accion


# Árbol visible (sólo profundidad 2 para claridad visual) 
class NodoArbol:
    # Nodo para dibujar el árbol Min-Max.
    def __init__(self, estado, accion=None, padre=None):
        self.estado  = estado
        self.accion  = accion
        self.padre   = padre
        self.hijos   = []
        self.valor   = None
        self.elegido = False


def construir_arbol(estado, profundidad=2):
    raiz = NodoArbol(estado)

    def expandir(nodo, prof, maximizando):
        if prof == 0 or nodo.estado.es_terminal():
            nodo.valor = evaluar_estado(nodo.estado)
            return
        acciones = ACCIONES_HEROE if maximizando else ACCIONES_JEFE
        for ac in acciones:
            nuevo = aplicar_accion(nodo.estado, ac)
            hijo = NodoArbol(nuevo, accion=ac, padre=nodo)
            nodo.hijos.append(hijo)
            expandir(hijo, prof - 1, not maximizando)

    expandir(raiz, profundidad, True)

    def propagar(nodo, maximizando):
        if not nodo.hijos:
            return
        for h in nodo.hijos:
            propagar(h, not maximizando)
        vals = [h.valor for h in nodo.hijos]
        nodo.valor = max(vals) if maximizando else min(vals)

    propagar(raiz, True)

    # marcar rama elegida
    def marcar(nodo, maximizando):
        if not nodo.hijos:
            return
        elegido = max(nodo.hijos, key=lambda h: h.valor) if maximizando else min(nodo.hijos, key=lambda h: h.valor)
        elegido.elegido = True
        marcar(elegido, not maximizando)

    marcar(raiz, True)
    return raiz


# Visualización del árbol 
def dibujar_arbol(raiz, filename="minmax_tree.png"):
    """
    Dibuja el árbol Min-Max con:
    - Nodos MAX (héroe) en verde, MIN (jefe) en coral
    - Rama elegida resaltada en ámbar grueso
    - Valor evaluado visible en cada nodo
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    aplicar_estilo_base(fig, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.1)
    ax.axis("off")
    ax.set_title("Árbol Min-Max — Decisión del héroe en batalla",
                 fontsize=14, fontweight="bold", color=COLORS["text"],
                 pad=14, fontfamily=FONT)

    # Calcular posiciones
    nodos_nivel = {}
    def asignar_pos(nodo, nivel=0, indice=0, total=1):
        nodos_nivel.setdefault(nivel, []).append(nodo)
        x = (indice + 0.5) / total
        y = 1.0 - nivel * 0.42
        nodo._pos = (x, y)
        for i, h in enumerate(nodo.hijos):
            asignar_pos(h, nivel + 1, i, len(nodo.hijos))

    asignar_pos(raiz)

    # Ajustar posiciones x por nivel
    for nivel, lista in nodos_nivel.items():
        n = len(lista)
        for i, nodo in enumerate(lista):
            nodo._pos = ((i + 0.5) / n, nodo._pos[1])

    # Dibujar aristas
    def dibujar_aristas(nodo):
        px, py = nodo._pos
        for hijo in nodo.hijos:
            hx, hy = hijo._pos
            color_linea = COLORS["chosen"] if hijo.elegido else "#C8C6BE"
            lw          = 3.0  if hijo.elegido else 1.0
            zorder      = 3    if hijo.elegido else 1
            ax.plot([px, hx], [py, hy],
                    color=color_linea, lw=lw, zorder=zorder,
                    solid_capstyle="round")
            # etiqueta de acción
            mx, my = (px + hx) / 2, (py + hy) / 2
            ax.text(mx + 0.01, my, hijo.accion,
                    fontsize=8, color=COLORS["text"],
                    ha="left", va="center",
                    fontfamily=FONT,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.75))
            dibujar_aristas(hijo)

    dibujar_aristas(raiz)

    # Dibujar nodos
    def dibujar_nodos(nodo, nivel=0):
        x, y = nodo._pos
        es_max   = (nivel % 2 == 0)
        es_hoja  = not nodo.hijos
        color_fondo = COLORS["hero"] if es_max else COLORS["enemy"]
        if es_hoja:
            color_fondo = "#D3D1C7"
        r = 0.042

        circulo = plt.Circle((x, y), r,
                              color=color_fondo, zorder=4)
        ax.add_patch(circulo)
        valor_str = str(int(nodo.valor)) if nodo.valor is not None else "?"
        ax.text(x, y, valor_str,
                fontsize=9, fontweight="bold",
                color="white", ha="center", va="center",
                zorder=5, fontfamily=FONT)
        # etiqueta MAX / MIN
        if nivel == 0:
            ax.text(x, y + r + 0.04, "MAX (héroe)",
                    fontsize=8, color=COLORS["hero"],
                    ha="center", fontfamily=FONT, fontweight="bold")
        elif not es_hoja and nivel == 1:
            ax.text(x, y + r + 0.04, "MIN (jefe)",
                    fontsize=8, color=COLORS["enemy"],
                    ha="center", fontfamily=FONT, fontweight="bold")
        for h in nodo.hijos:
            dibujar_nodos(h, nivel + 1)

    dibujar_nodos(raiz)

    # Leyenda
    leyenda = [
        mpatches.Patch(color=COLORS["hero"],   label="Nodo MAX — héroe elige"),
        mpatches.Patch(color=COLORS["enemy"],  label="Nodo MIN — jefe responde"),
        mpatches.Patch(color="#D3D1C7",        label="Hoja — valor evaluado"),
        mpatches.Patch(color=COLORS["chosen"], label="Rama elegida por el agente"),
    ]
    ax.legend(handles=leyenda, loc="lower center", ncol=4,
              fontsize=8, frameon=True,
              facecolor=COLORS["bg"], edgecolor="#D3D1C7")

    plt.tight_layout()
    plt.savefig(ruta(filename), dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  [OK] {ruta(filename)} guardado.")



#  MÓDULO 2 — STRIPS

#Estado: diccionario de proposiciones booleanas
# Acciones: nombre, precondiciones, efectos_agregar, efectos_quitar

ACCIONES_STRIPS = [
    {
        "nombre":   "Comprar_Espada",
        "pre":      {"en_aldea": True,  "tiene_espada": False},
        "add":      {"tiene_espada": True},
        "del":      [],
    },
    {
        "nombre":   "Ir_Mazmorra",
        "pre":      {"en_aldea": True,  "tiene_espada": True},
        "add":      {"en_mazmorra": True, "en_aldea": False},
        "del":      [],
    },
    {
        "nombre":   "Debilitar_Jefe",
        "pre":      {"en_mazmorra": True, "jefe_fuerte": True},
        "add":      {"jefe_debil": True, "jefe_fuerte": False},
        "del":      [],
    },
    {
        "nombre":   "Atacar_Jefe_Final",
        "pre":      {"en_mazmorra": True, "jefe_debil": True, "tiene_espada": True},
        "add":      {"jefe_derrotado": True, "jefe_debil": False},
        "del":      [],
    },
]

ESTADO_INICIAL_STRIPS = {
    "en_aldea":      True,
    "tiene_espada":  False,
    "en_mazmorra":   False,
    "jefe_fuerte":   True,
    "jefe_debil":    False,
    "jefe_derrotado": False,
}

META_STRIPS = {"jefe_derrotado": True}


def estado_satisface_meta(estado, meta):
    return all(estado.get(k) == v for k, v in meta.items())


def aplicar_accion_strips(estado, accion):
    #Aplica una acción STRIPS al estado y devuelve el nuevo estado.
    if not all(estado.get(k) == v for k, v in accion["pre"].items()):
        return None
    nuevo = dict(estado)
    nuevo.update(accion["add"])
    for k in accion["del"]:
        nuevo[k] = False
    return nuevo


def planificar_strips(estado_ini, meta, acciones, max_iter=50):
    """
    BFS simple sobre el espacio de estados STRIPS.
    Devuelve (plan, todos_los_estados) donde plan es la lista de acciones.
    """
    from collections import deque
    cola = deque()
    inicio = frozenset((k, v) for k, v in estado_ini.items())
    cola.append((inicio, []))
    visitados = {inicio}
    todos_los_estados = [estado_ini]
    transiciones = []

    for _ in range(max_iter):
        if not cola:
            break
        estado_fs, plan = cola.popleft()
        estado_dict = dict(estado_fs)

        if estado_satisface_meta(estado_dict, meta):
            return plan, todos_los_estados, transiciones

        for accion in acciones:
            nuevo = aplicar_accion_strips(estado_dict, accion)
            if nuevo is None:
                continue
            nuevo_fs = frozenset((k, v) for k, v in nuevo.items())
            if nuevo_fs not in visitados:
                visitados.add(nuevo_fs)
                todos_los_estados.append(nuevo)
                transiciones.append((estado_dict, nuevo, accion["nombre"]))
                cola.append((nuevo_fs, plan + [accion["nombre"]]))

    return None, todos_los_estados, transiciones


def nombre_corto_estado(estado):
    # Devuelve una etiqueta corta y legible para el estado.
    if estado.get("jefe_derrotado"):
        return "Jefe\nderrotado"
    if estado.get("en_mazmorra") and estado.get("jefe_debil"):
        return "Mazmorra\njefe débil"
    if estado.get("en_mazmorra"):
        return "Mazmorra\njefe fuerte"
    if estado.get("en_aldea") and estado.get("tiene_espada"):
        return "Aldea\ncon espada"
    return "Aldea\nsin espada"


def dibujar_strips(plan, transiciones, filename="strips_graph.png"):
    """
    Dibuja el grafo de estados STRIPS con:
    - Estado inicial en azul, meta en verde
    - Ruta del plan resaltada
    - Leyenda clara
    """
    G = nx.DiGraph()
    estados_plan_labels = set()

    # Reconstruir la ruta del plan para resaltarla
    estado_actual = dict(ESTADO_INICIAL_STRIPS)
    estados_plan_labels.add(nombre_corto_estado(estado_actual))
    for nombre_ac in plan:
        for ac in ACCIONES_STRIPS:
            if ac["nombre"] == nombre_ac:
                estado_actual = aplicar_accion_strips(estado_actual, ac)
                estados_plan_labels.add(nombre_corto_estado(estado_actual))
                break

    # Construir grafo
    for ori, dst, nombre_ac in transiciones:
        etq_ori = nombre_corto_estado(ori)
        etq_dst = nombre_corto_estado(dst)
        G.add_edge(etq_ori, etq_dst, label=nombre_ac.replace("_", " "))

    inicio_label = nombre_corto_estado(ESTADO_INICIAL_STRIPS)
    meta_label   = "Jefe\nderrotado"

    fig, ax = plt.subplots(figsize=(12, 6))
    aplicar_estilo_base(fig, ax)
    ax.set_facecolor(COLORS["bg"])
    ax.set_title("Planificador STRIPS — Ruta para derrotar al Jefe Final",
                 fontsize=14, fontweight="bold", color=COLORS["text"],
                 pad=14, fontfamily=FONT)
    ax.axis("off")

    pos = nx.spring_layout(G, seed=7, k=2.5)

    # Colorear nodos
    nodo_colores = []
    for nodo in G.nodes():
        if nodo == inicio_label:
            nodo_colores.append(COLORS["plan"])
        elif nodo == meta_label:
            nodo_colores.append(COLORS["hero"])
        elif nodo in estados_plan_labels:
            nodo_colores.append("#B5D4F4")
        else:
            nodo_colores.append(COLORS["light"])

    nodo_border = []
    for nodo in G.nodes():
        if nodo in estados_plan_labels:
            nodo_border.append(2.5)
        else:
            nodo_border.append(0.8)

    # Colorear aristas
    aristas_plan = set()
    estado_ruta = dict(ESTADO_INICIAL_STRIPS)
    for nombre_ac in plan:
        for ac in ACCIONES_STRIPS:
            if ac["nombre"] == nombre_ac:
                etq_ori = nombre_corto_estado(estado_ruta)
                estado_ruta = aplicar_accion_strips(estado_ruta, ac)
                etq_dst = nombre_corto_estado(estado_ruta)
                aristas_plan.add((etq_ori, etq_dst))
                break

    edge_colors = [COLORS["chosen"] if (u, v) in aristas_plan else "#D3D1C7"
                   for u, v in G.edges()]
    edge_widths = [3.5 if (u, v) in aristas_plan else 1.0
                   for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=nodo_colores,
                           node_size=2200,
                           linewidths=nodo_border,
                           edgecolors=[COLORS["plan"] if n in estados_plan_labels else "#B4B2A9" for n in G.nodes()])

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=edge_widths,
                           arrows=True,
                           arrowsize=20,
                           connectionstyle="arc3,rad=0.1",
                           node_size=2200)

    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=9, font_color=COLORS["text"],
                            font_family=FONT, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=8,
                                 font_color=COLORS["text"],
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           fc="white", ec="none", alpha=0.8))

    # Leyenda
    leyenda = [
        mpatches.Patch(color=COLORS["plan"],   label="Estado inicial"),
        mpatches.Patch(color=COLORS["hero"],   label="Meta (jefe derrotado)"),
        mpatches.Patch(color="#B5D4F4",        label="Estado en el plan"),
        mpatches.Patch(color=COLORS["light"],  label="Estado explorado"),
        mpatches.Patch(color=COLORS["chosen"], label="Ruta del plan"),
    ]
    ax.legend(handles=leyenda, loc="lower center", ncol=5,
              fontsize=8, frameon=True,
              facecolor=COLORS["bg"], edgecolor="#D3D1C7")

    plt.tight_layout()
    plt.savefig(ruta(filename), dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  [OK] {ruta(filename)} guardado.")


#  MÓDULO 3 — RED BAYESIANA

"""
Estructura de la red:
  P(Enemigo_Peligroso)          — prior configurable
  P(HP_Bajo | —)                — prior fijo
  P(Tiene_Arma | —)             — prior fijo
  P(Victoria | Enemigo, HP, Arma) — tabla de probabilidades condicionales
"""

# Probabilidades a priori
P_ENEMIGO_PELIGROSO = 0.40
P_HP_BAJO           = 0.35  # héroe con vida baja
P_TIENE_ARMA        = 0.75

# Tabla CPT: P(Victoria=True | Enemigo, HP_Bajo, Tiene_Arma)
# Clave: (enemigo_peligroso, hp_bajo, tiene_arma)
CPT_VICTORIA = {
    (True,  True,  True):  0.35,
    (True,  True,  False): 0.10,
    (True,  False, True):  0.60,
    (True,  False, False): 0.25,
    (False, True,  True):  0.70,
    (False, True,  False): 0.40,
    (False, False, True):  0.90,
    (False, False, False): 0.55,
}


def calcular_posterior(evidencia: dict, p_enemigo_peligroso=None) -> float:
    """
    Calcula P(Victoria=True | evidencia) por enumeración.
    evidencia puede contener cualquier subconjunto de
    {enemigo_peligroso, hp_bajo, tiene_arma}.
    """
    if p_enemigo_peligroso is None:
        p_enemigo_peligroso = P_ENEMIGO_PELIGROSO

    p_priors = {
        "enemigo_peligroso": p_enemigo_peligroso,
        "hp_bajo":           P_HP_BAJO,
        "tiene_arma":        P_TIENE_ARMA,
    }

    numerador   = 0.0
    denominador = 0.0
    variables   = ["enemigo_peligroso", "hp_bajo", "tiene_arma"]

    for vals in itertools.product([True, False], repeat=3):
        asignacion = dict(zip(variables, vals))

        # Consistente con evidencia
        if not all(asignacion[k] == v for k, v in evidencia.items()
                   if k in asignacion):
            continue

        # P(variables)
        p_vars = 1.0
        for var, val in asignacion.items():
            p_var = p_priors[var]
            p_vars *= p_var if val else (1 - p_var)

        clave = tuple(asignacion[v] for v in variables)
        p_victoria = CPT_VICTORIA[clave]

        numerador   += p_victoria * p_vars
        denominador += p_vars

    return numerador / denominador if denominador > 0 else 0.0


def dibujar_bayes_bars(evidencia: dict, filename="bayes_bars.png"):
    """
    Gráfico de barras horizontal con las probabilidades bayesianas.
    Muestra P(Victoria) y P(Derrota) con la evidencia usada.
    """
    p_vic = calcular_posterior(evidencia)
    p_der = 1 - p_vic

    etiquetas = ["P(Victoria)", "P(Derrota)"]
    valores   = [p_vic, p_der]
    colores   = [COLORS["hero"], COLORS["enemy"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    aplicar_estilo_base(fig, ax)
    ax.set_facecolor(COLORS["bg"])

    bars = ax.barh(etiquetas, valores, color=colores,
                   height=0.45, edgecolor="none")

    # Valores al final de cada barra
    for bar, val in zip(bars, valores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}",
                va="center", ha="left",
                fontsize=16, fontweight="bold",
                color=COLORS["text"], fontfamily=FONT)

    # Línea de referencia 50 %
    ax.axvline(0.5, color=COLORS["neutral"], linewidth=1.2,
               linestyle="--", label="Umbral 50 %")

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Probabilidad", fontsize=11,
                  color=COLORS["text"], fontfamily=FONT)
    ax.set_title("Red Bayesiana — Probabilidad de victoria en batalla",
                 fontsize=13, fontweight="bold",
                 color=COLORS["text"], pad=12, fontfamily=FONT)

    # Caja de evidencia dentro del gráfico
    desc_ev = []
    mapeo = {
        "enemigo_peligroso": ("Enemigo peligroso", "Sí", "No"),
        "hp_bajo":           ("HP del héroe",       "Bajo", "Alto"),
        "tiene_arma":        ("Tiene arma",          "Sí", "No"),
    }
    for k, v in evidencia.items():
        nombre, si, no = mapeo.get(k, (k, "True", "False"))
        desc_ev.append(f"{nombre}: {si if v else no}")
    texto_ev = "Evidencia usada:\n" + "\n".join(desc_ev)
    ax.text(0.97, 0.5, texto_ev,
            transform=ax.transAxes,
            fontsize=9, va="center", ha="right",
            fontfamily=FONT,
            color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.5",
                      fc=COLORS["light"], ec="#B4B2A9", alpha=0.9))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(ruta(filename), dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  [OK] {ruta(filename)} guardado.")



#  MÓDULO 4 — AGENTE INTEGRADOR

def dibujar_agent_summary(plan, p_victoria, mejor_accion,
                          estado, filename="agent_summary.png"):
    """
    Figura resumen del agente con 3 paneles:
      1. Plan STRIPS (lista visual)
      2. Probabilidad de victoria (mini barra)
      3. Decisión final Min-Max (texto grande)
    """
    fig = plt.figure(figsize=(15, 6))
    aplicar_estilo_base(fig)
    fig.suptitle("Resumen del agente — Turno actual",
                 fontsize=16, fontweight="bold",
                 color=COLORS["text"], fontfamily=FONT, y=1.0)

    # ── Panel 1: Plan STRIPS ──
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor(COLORS["bg"])
    ax1.set_title("Plan STRIPS", fontsize=12, fontweight="bold",
                  color=COLORS["plan"], fontfamily=FONT, pad=10)
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    n = len(plan)
    for i, paso in enumerate(plan):
        y = 0.85 - i * (0.65 / max(n - 1, 1))
        color_caja = COLORS["plan"] if i == 0 else (COLORS["hero"] if i == n - 1 else "#B5D4F4")
        color_text = "white" if i in (0, n - 1) else COLORS["text"]
        rect = FancyBboxPatch((0.08, y - 0.065), 0.84, 0.11,
                              boxstyle="round,pad=0.01",
                              facecolor=color_caja, edgecolor="none")
        ax1.add_patch(rect)
        ax1.text(0.5, y - 0.01, f"{i+1}. {paso.replace('_', ' ')}",
                 ha="center", va="center", fontsize=9,
                 fontweight="bold", color=color_text, fontfamily=FONT)
        if i < n - 1:
            ax1.annotate("", xy=(0.5, y - 0.075),
                         xytext=(0.5, y - 0.065 - 0.065),
                         arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=1.5))

    ax1.text(0.5, 0.04, f"{n} pasos hasta derrotar al jefe",
             ha="center", va="bottom", fontsize=8,
             color=COLORS["neutral"], fontfamily=FONT)

    # ── Panel 2: Probabilidad Bayesiana ──
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor(COLORS["bg"])
    ax2.set_title("Probabilidad de victoria", fontsize=12, fontweight="bold",
                  color=COLORS["bayes"], fontfamily=FONT, pad=10)

    p_der = 1 - p_victoria
    etiquetas = ["Victoria", "Derrota"]
    valores   = [p_victoria, p_der]
    colores   = [COLORS["hero"], COLORS["enemy"]]

    bars = ax2.bar(etiquetas, valores, color=colores, width=0.45, edgecolor="none")
    for bar, val in zip(bars, valores):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.02, f"{val:.0%}",
                 ha="center", va="bottom",
                 fontsize=18, fontweight="bold",
                 color=COLORS["text"], fontfamily=FONT)

    ax2.axhline(0.5, color=COLORS["neutral"], linewidth=1.2,
                linestyle="--", alpha=0.7, label="Umbral 50 %")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Probabilidad", fontsize=10,
                   color=COLORS["text"], fontfamily=FONT)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", labelsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.set_facecolor(COLORS["bg"])
    ax2.legend(fontsize=8, frameon=False)

    # ── Panel 3: Decisión Min-Max ──
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(COLORS["bg"])
    ax3.set_title("Decisión Min-Max", fontsize=12, fontweight="bold",
                  color=COLORS["minmax"], fontfamily=FONT, pad=10)
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Cuadro grande con la acción elegida
    rect_dec = FancyBboxPatch((0.08, 0.52), 0.84, 0.32,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS["chosen"], edgecolor="none")
    ax3.add_patch(rect_dec)
    ax3.text(0.5, 0.68, mejor_accion.upper(),
             ha="center", va="center",
             fontsize=22, fontweight="bold",
             color="white", fontfamily=FONT)
    ax3.text(0.5, 0.56, "Acción recomendada",
             ha="center", va="center",
             fontsize=9, color="white", fontfamily=FONT)

    # Estado actual
    ax3.text(0.5, 0.44, "Estado actual:",
             ha="center", va="center", fontsize=9,
             color=COLORS["neutral"], fontfamily=FONT)
    ax3.text(0.5, 0.36, f"HP héroe: {estado.hp_heroe}  |  HP jefe: {estado.hp_jefe}",
             ha="center", va="center", fontsize=10,
             color=COLORS["text"], fontfamily=FONT, fontweight="bold")
    ax3.text(0.5, 0.28, f"Ítems disponibles: {estado.items}",
             ha="center", va="center", fontsize=10,
             color=COLORS["text"], fontfamily=FONT)

    # Indicador de confianza
    confianza = "Alta" if p_victoria > 0.65 else ("Media" if p_victoria > 0.40 else "Baja")
    color_conf = COLORS["hero"] if p_victoria > 0.65 else (COLORS["chosen"] if p_victoria > 0.40 else COLORS["enemy"])
    ax3.text(0.5, 0.15,
             f"Confianza bayesiana: {confianza} ({p_victoria:.0%})",
             ha="center", va="center", fontsize=10,
             color=color_conf, fontfamily=FONT, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(ruta(filename), dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  [OK] {ruta(filename)} guardado.")

#  EXPERIMENTOS: Comparación Min-Max

ESCENARIOS = {
    "Ventaja héroe":    EstadoJuego(hp_heroe=80, hp_jefe=30, items=2),
    "Empate":           EstadoJuego(hp_heroe=60, hp_jefe=60, items=1),
    "Desventaja héroe": EstadoJuego(hp_heroe=30, hp_jefe=80, items=0),
}
PROFUNDIDADES = [2, 3, 4]


def correr_experimentos():
    """
    Corre los 3 algoritmos × 3 escenarios × 3 profundidades.
    Devuelve lista de dicts con los resultados.
    """
    resultados = []
    for nombre_esc, estado_ini in ESCENARIOS.items():
        for prof in PROFUNDIDADES:
            fila = {"escenario": nombre_esc, "prof": prof}

            global _nodos_naive, _nodos_ab, _nodos_heur

            # Naive
            _nodos_naive = 0
            t0 = time.perf_counter()
            v_n, ac_n = minmax_naive(estado_ini, prof, True)
            fila["t_naive"]      = round(time.perf_counter() - t0, 4)
            fila["nodos_naive"]  = _nodos_naive
            fila["dec_naive"]    = ac_n

            # α-β
            _nodos_ab = 0
            t0 = time.perf_counter()
            v_ab, ac_ab = minmax_ab(estado_ini, prof, -math.inf, math.inf, True)
            fila["t_ab"]         = round(time.perf_counter() - t0, 4)
            fila["nodos_ab"]     = _nodos_ab
            fila["dec_ab"]       = ac_ab

            # Heurística
            _nodos_heur = 0
            t0 = time.perf_counter()
            v_h, ac_h = minmax_heur(estado_ini, prof, -math.inf, math.inf, True)
            fila["t_heur"]       = round(time.perf_counter() - t0, 4)
            fila["nodos_heur"]   = _nodos_heur
            fila["dec_heur"]     = ac_h

            resultados.append(fila)
    return resultados



#  ANÁLISIS DE SENSIBILIDAD BAYESIANO

PRIORS_SENSIBILIDAD = [0.05, 0.10, 0.20, 0.40, 0.60]
EVIDENCIA_FIJA = {"hp_bajo": False, "tiene_arma": True}


def analisis_sensibilidad():
    """
    Varía P(Enemigo_Peligroso) en 5 valores.
    Para cada prior calcula: posterior, impacto en plan y en decisión Min-Max.
    """
    resultados = []
    for p_prior in PRIORS_SENSIBILIDAD:
        p_post = calcular_posterior(EVIDENCIA_FIJA, p_enemigo_peligroso=p_prior)

        # El umbral para seguir el plan es P(Victoria) > 0.5
        seguir_plan = p_post > 0.50

        # Ajustar estado para Min-Max según el prior
        hp_jefe = int(30 + p_prior * 70)  # más peligroso → más HP
        estado_sa = EstadoJuego(hp_heroe=65, hp_jefe=hp_jefe, items=1)
        _, decision_mm = minmax_heur(estado_sa, 3, -math.inf, math.inf, True)

        resultados.append({
            "prior":       p_prior,
            "posterior":   p_post,
            "seguir_plan": seguir_plan,
            "decision_mm": decision_mm,
        })
    return resultados


def dibujar_sensibilidad(resultados_sens, filename="sensitivity_analysis.png"):
    # Gráfico de sensibilidad bayesiana.
    priors     = [r["prior"] for r in resultados_sens]
    posteriors = [r["posterior"] for r in resultados_sens]
    decisions  = [r["decision_mm"] for r in resultados_sens]

    fig, ax = plt.subplots(figsize=(10, 5))
    aplicar_estilo_base(fig, ax)
    ax.set_facecolor(COLORS["bg"])

    ax.plot(priors, posteriors,
            color=COLORS["bayes"], linewidth=2.5,
            marker="o", markersize=9,
            markerfacecolor=COLORS["bayes"], label="P(Victoria | evidencia)")
    ax.axhline(0.5, color=COLORS["neutral"], linestyle="--",
               linewidth=1.2, label="Umbral de decisión 50 %")

    # Anotaciones de decisión Min-Max
    for p, post, dec in zip(priors, posteriors, decisions):
        ax.annotate(dec, xy=(p, post),
                    xytext=(p, post + 0.07),
                    fontsize=8, ha="center",
                    color=COLORS["minmax"], fontfamily=FONT,
                    arrowprops=dict(arrowstyle="-", color=COLORS["neutral"],
                                   lw=0.8))

    # Regiones
    ax.axhspan(0.5, 1.0, alpha=0.06, color=COLORS["hero"], label="Zona: plan activo")
    ax.axhspan(0.0, 0.5, alpha=0.06, color=COLORS["enemy"], label="Zona: plan en riesgo")

    ax.set_xlabel("Prior P(Enemigo Peligroso)", fontsize=11,
                  color=COLORS["text"], fontfamily=FONT)
    ax.set_ylabel("Posterior P(Victoria)", fontsize=11,
                  color=COLORS["text"], fontfamily=FONT)
    ax.set_title("Análisis de sensibilidad — Impacto del prior en la victoria",
                 fontsize=13, fontweight="bold",
                 color=COLORS["text"], pad=12, fontfamily=FONT)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.02, 0.65)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=True,
              facecolor=COLORS["bg"])

    plt.tight_layout()
    plt.savefig(ruta(filename), dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  [OK] {ruta(filename)} guardado.")


#  LOG DEL AGENTE

class AgentLog:
    def __init__(self, filename="agent_log.txt"):
        self.filename = filename
        self.lineas   = []

    def log(self, msg):
        print(f"  {msg}")
        self.lineas.append(msg)

    def guardar(self):
        with open(ruta(self.filename), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lineas))
        print(f"  [OK] {self.filename} guardado.")

#  MAIN — EJECUTAR TODO

def main():
    log = AgentLog()
    log.log("=" * 60)
    log.log("  AGENTE IA — BATALLA RPG POR TURNOS")
    log.log("  Sector: Videojuegos | Parcial Corte 2")
    log.log("=" * 60)

    # ── Estado inicial del juego ──
    estado = EstadoJuego(hp_heroe=65, hp_jefe=40, items=2)
    log.log(f"\n[OBSERVAR] Estado actual: {estado}")

    # ── Paso 1: Red Bayesiana ──
    log.log("\n─── MÓDULO 1: Red Bayesiana ───")
    evidencia = {
        "enemigo_peligroso": False,
        "hp_bajo":           False,
        "tiene_arma":        True,
    }
    p_victoria = calcular_posterior(evidencia)
    log.log(f"  Evidencia: {evidencia}")
    log.log(f"  P(Victoria | evidencia) = {p_victoria:.4f} ({p_victoria:.0%})")
    log.log(f"  Decisión bayesiana: {'SEGUIR PLAN' if p_victoria > 0.5 else 'REPLANTEAR ESTRATEGIA'}")

    print("\n[1/5] Generando bayes_bars.png...")
    dibujar_bayes_bars(evidencia)

    # ── Paso 2: STRIPS ──
    log.log("\n─── MÓDULO 2: Planificador STRIPS ───")
    plan, todos_estados, transiciones = planificar_strips(
        ESTADO_INICIAL_STRIPS, META_STRIPS, ACCIONES_STRIPS
    )
    if plan:
        log.log(f"  Plan encontrado ({len(plan)} pasos):")
        for i, paso in enumerate(plan, 1):
            log.log(f"    {i}. {paso}")
    else:
        log.log("  No se encontró plan.")
        plan = ["Atacar_Jefe_Final"]

    print("\n[2/5] Generando strips_graph.png...")
    dibujar_strips(plan, transiciones)

    # ── Paso 3: Min-Max ──
    log.log("\n─── MÓDULO 3: Min-Max (profundidad 3) ───")
    global _nodos_ab
    _nodos_ab = 0
    valor_mm, mejor_accion = minmax_ab(estado, 3, -math.inf, math.inf, True)
    log.log(f"  Mejor acción (α-β): {mejor_accion} (valor={valor_mm:.1f})")
    log.log(f"  Nodos expandidos:   {_nodos_ab}")

    print("\n[3/5] Generando minmax_tree.png...")
    raiz = construir_arbol(estado, profundidad=2)
    dibujar_arbol(raiz)

    # ── Paso 4: Agente integrador ──
    log.log("\n─── MÓDULO 4: Agente integrador ───")
    log.log(f"  observe  → {estado}")
    log.log(f"  posterior→ P(Victoria) = {p_victoria:.0%}")
    log.log(f"  plan     → {' → '.join(plan)}")
    log.log(f"  decision → {mejor_accion}")
    log.log(f"  acción   → EJECUTAR: {mejor_accion}")

    print("\n[4/5] Generando agent_summary.png...")
    dibujar_agent_summary(plan, p_victoria, mejor_accion, estado)

    # ── Paso 5: Experimentos de comparación ──
    log.log("\n─── EXPERIMENTOS: Comparación de versiones Min-Max ───")
    resultados_exp = correr_experimentos()
    log.log(f"\n  {'Escenario':<22} {'Prof':>4} | {'Naive(ms)':>9} {'Nodos':>6} | {'α-β(ms)':>7} {'Nodos':>6} | {'Heur(ms)':>8} {'Nodos':>6} | Decisión")
    log.log("  " + "-" * 110)
    for r in resultados_exp:
        log.log(
            f"  {r['escenario']:<22} {r['prof']:>4} | "
            f"{r['t_naive']*1000:>8.2f} {r['nodos_naive']:>6} | "
            f"{r['t_ab']*1000:>6.2f} {r['nodos_ab']:>6} | "
            f"{r['t_heur']*1000:>7.2f} {r['nodos_heur']:>6} | "
            f"{r['dec_naive']} / {r['dec_ab']} / {r['dec_heur']}"
        )

    # ── Paso 6: Análisis de sensibilidad ──
    log.log("\n─── ANÁLISIS DE SENSIBILIDAD BAYESIANO ───")
    resultados_sens = analisis_sensibilidad()
    log.log(f"\n  Evidencia fija: HP alto, tiene arma")
    log.log(f"  {'Prior(Peligro)':>15} | {'P(Victoria)':>11} | {'Seguir plan?':>12} | Decisión MinMax")
    log.log("  " + "-" * 62)
    for r in resultados_sens:
        log.log(
            f"  {r['prior']:>14.0%} | "
            f"{r['posterior']:>10.0%} | "
            f"{'Sí' if r['seguir_plan'] else 'No':>12} | "
            f"{r['decision_mm']}"
        )

    dibujar_sensibilidad(resultados_sens)

    # ── Guardar log ──
    log.log("\n" + "=" * 60)
    log.log("  Todos los archivos generados correctamente.")
    log.log("=" * 60)
    log.guardar()

    print("\n" + "=" * 60)
    print("  Archivos generados:")
    print("    minmax_tree.png")
    print("    strips_graph.png")
    print("    bayes_bars.png")
    print("    agent_summary.png")
    print("    sensitivity_analysis.png")
    print("    agent_log.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
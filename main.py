import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import matplotlib.image as mpimg
from PIL import Image
import io

# --- 1. Definições e Inicializações ---

estados = [
    [0, 1, 2, 3],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25, 26, 27],
    [28, 29, 30, 31, 32, 33, 34, 35]
]

coords = {
    0: (2, 0), 1: (3, 0), 2: (4, 0), 3: (5, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1), 7: (3, 1), 8: (4, 1), 9: (5, 1), 10: (6, 1), 11: (7, 1),
    12: (0, 2), 13: (1, 2), 14: (2, 2), 15: (3, 2), 16: (4, 2), 17: (5, 2), 18: (6, 2), 19: (7, 2),
    20: (0, 3), 21: (1, 3), 22: (2, 3), 23: (3, 3), 24: (4, 3), 25: (5, 3), 26: (6, 3), 27: (7, 3),
    28: (0, 4), 29: (1, 4), 30: (2, 4), 31: (3, 4), 32: (4, 4), 33: (5, 4), 34: (6, 4), 35: (7, 4),
}

# Criar dados de exemplo para a matriz de distâncias
# Vamos usar valores específicos para testar o cálculo
indices = [f'E{i+1}' for i in range(36)]
df_dist = pd.DataFrame(np.ones((36, 36)), index=indices, columns=indices)

# Definir valores específicos para o teste
df_dist.loc['E15', 'E7'] = 0.5  # Custo de E15 para E7
df_dist.loc['E15', 'E6'] = 0.5  # Custo de E15 para E6
df_dist.loc['E15', 'E14'] = 0.5  # Custo de E15 para E14
df_dist.loc['E15', 'E22'] = 0.5  # Custo de E15 para E22
df_dist.loc['E15', 'E23'] = 0.5  # Custo de E15 para E23

df_dist.loc['E7', 'E1'] = 0.5  # Custo de E7 para E1
df_dist.loc['E7', 'E6'] = 0.5  # Custo de E7 para E6
df_dist.loc['E7', 'E14'] = 0.5  # Custo de E7 para E14

# Valores de distância para o objetivo (E9)
df_dist.loc['E7', 'E9'] = 1.2  # Distância de E7 para E9
df_dist.loc['E6', 'E9'] = 1.8  # Distância de E6 para E9
df_dist.loc['E14', 'E9'] = 1.87  # Distância de E14 para E9
df_dist.loc['E22', 'E9'] = 2.06  # Distância de E22 para E9
df_dist.loc['E23', 'E9'] = 1.56  # Distância de E23 para E9
df_dist.loc['E1', 'E9'] = 1.3  # Distância de E1 para E9
df_dist.loc['E15', 'E9'] = 0.697  # Distância de E15 para E9

# Garantir que a matriz seja simétrica
for i in range(36):
    for j in range(36):
        if i != j:
            df_dist.iloc[i, j] = df_dist.iloc[j, i]

def distancia(estado1, estado2):
    return df_dist.loc[f'E{estado1+1}', f'E{estado2+1}']

PESO_B = 1.1
PESO_R = 1.1

adversarios = np.zeros(36, dtype=int)
aliados = np.zeros(36, dtype=int)

# Configuração do cenário do seu exemplo
estado_inicial = 14  # E15
estado_objetivo = 8  # E9
estado_bola = 35     # E36
aliados[24] = 1 
adversarios[7] = 1   # E8
adversarios[9] = 1   # E10
adversarios[15] = 1  # E16
adversarios[29] = 1  # E30

def vizinhos(estado):
    x, y = coords[estado]
    viz = []
    for e, (ex, ey) in coords.items():
        dx, dy = ex - x, ey - y
        if (dx, dy) == (0, 0):
            continue
        if abs(dx) <= 1 and abs(dy) <= 1:
            # Se for movimento diagonal, checar bloqueio lateral
            if abs(dx) == 1 and abs(dy) == 1:
                # Índices dos estados laterais
                lateral1 = None
                lateral2 = None
                for k, (kx, ky) in coords.items():
                    if (kx, ky) == (x + dx, y):
                        lateral1 = k
                    if (kx, ky) == (x, y + dy):
                        lateral2 = k
                if (lateral1 is not None and adversarios[lateral1]) or (lateral2 is not None and adversarios[lateral2]):
                    continue  # Não permite diagonal se houver adversário lateral
            if not adversarios[e] and not aliados[e]:
                viz.append(e)
    return viz

def penalizacao_B(estado, estado_bola):
    xi, yi = coords[estado]
    min_dist_estado = distancia(estado, estado_bola)
    for e in range(36):
        if adversarios[e]:
            xe, ye = coords[e]
            # Adversário na mesma linha ou coluna
            if xi == xe or yi == ye:
                dist_adv = distancia(e, estado_bola)
                if dist_adv < min_dist_estado:
                    return PESO_B
    return 1.0

def penalizacao_R(estado):
    x, y = coords[estado]
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            for e, (ex, ey) in coords.items():
                if (ex, ey) == (nx, ny):
                    if adversarios[e]:
                        count += 1
    if count >= 2:
        return PESO_R
    return 1.0

def heuristica(estado, estado_objetivo, estado_bola):
    d = distancia(estado, estado_objetivo)
    B = penalizacao_B(estado, estado_bola)
    R = penalizacao_R(estado)
    h = d * B * R
    return h, d, B, R

# CORREÇÃO: Função A* corrigida para calcular f(x) = g(x) + h(x) corretamente
def a_star(estado_inicial, estado_objetivo, estado_bola):
    open_set = []
    heapq.heappush(open_set, (0, estado_inicial))
    came_from = {}
    g_score = {estado_inicial: 0}
    h, d, B, R = heuristica(estado_inicial, estado_objetivo, estado_bola)
    f_score = {estado_inicial: h}
    detalhes = {estado_inicial: {'g': 0, 'h': h, 'd': d, 'B': B, 'R': R, 'f': h}}

    primeira_expansao = True
    expansoes = []

    while open_set:
        _, atual = heapq.heappop(open_set)
        
        # Registrar expansão para debug
        if atual not in expansoes:
            expansoes.append(atual)
            
            # Mostrar detalhes da expansão
            print(f"\nExpansão de E{atual+1}:")
            vizs = vizinhos(atual)
            for viz in vizs:
                custo = distancia(atual, viz)
                g = g_score[atual] + custo
                h, d, B, R = heuristica(viz, estado_objetivo, estado_bola)
                f = g + h
                print(f"  E{viz+1}: g={g:.3f}, h={h:.3f} (d={d:.3f}, B={B:.2f}, R={R:.2f}), f={f:.3f}")
        
        if primeira_expansao:
            print("\nVizinhos de E15 e seus valores de h(x):")
            for viz in vizinhos(atual):
                h, d, B, R = heuristica(viz, estado_objetivo, estado_bola)
                print(f"E{viz+1}: d={d:.3f}, B={B:.2f}, R={R:.2f}, h={h:.3f}")
            primeira_expansao = False

        if atual == estado_objetivo:
            caminho = [atual]
            while atual in came_from:
                atual = came_from[atual]
                caminho.append(atual)
            caminho.reverse()
            return caminho, detalhes
            
        for viz in vizinhos(atual):
            custo = distancia(atual, viz)
            g = g_score[atual] + custo  # Custo acumulativo
            h, d, B, R = heuristica(viz, estado_objetivo, estado_bola)
            f = g + h  # f(x) = g(x) + h(x)
            
            if viz not in g_score or g < g_score[viz]:
                came_from[viz] = atual
                g_score[viz] = g
                f_score[viz] = f
                detalhes[viz] = {'g': g, 'h': h, 'd': d, 'B': B, 'R': R, 'f': f}
                heapq.heappush(open_set, (f, viz))
                
    return None, {}

# Criar imagens de exemplo para os robôs
def create_dummy_image(color, text=""):
    img = np.ones((100, 100, 4))  # RGBA
    img[:, :, 0] = color[0]  # R
    img[:, :, 1] = color[1]  # G
    img[:, :, 2] = color[2]  # B
    img[:, :, 3] = 1.0  # Alpha (opacidade)
    return img

# Criar imagens de exemplo
img_goleiro = create_dummy_image([0, 0, 1])  # Azul
img_adversario = create_dummy_image([1, 0.5, 0.7])  # Rosa
img_adversario_bola = create_dummy_image([1, 0.3, 0.5])  # Rosa mais escuro
img_aliado = create_dummy_image([0.5, 0.5, 1])  # Azul claro

def plot_campo(caminho, estado_bola):
    fig, ax = plt.subplots(figsize=(12, 8))

    # IMPORTANTE: Marcar o adversário na posição da bola
    adversarios[estado_bola] = 1  # Garantir que há um adversário na posição da bola

    for e in range(36):
        x, y = coords[e]
        # Fundo da célula - CORRIGIDO: usar -y-1 para inverter corretamente
        rect = plt.Rectangle((x, -y-1), 1, 1, facecolor='white', edgecolor='k', zorder=1)
        ax.add_patch(rect)

        # Objetivo destacado (verde)
        if e == estado_objetivo:
            rect = plt.Rectangle((x, -y-1), 1, 1, facecolor='green', edgecolor='k', zorder=2)
            ax.add_patch(rect)

        # Sobreponha a imagem se houver robô - CORRIGIDO: extent ajustado
        if e == estado_inicial:
            ax.imshow(img_goleiro, extent=(x+0.1, x+0.9, -y-0.1, -y-0.9), zorder=3)
        elif e == estado_bola:
            ax.imshow(img_adversario_bola, extent=(x+0.1, x+0.9, -y-0.1, -y-0.9), zorder=3)
        elif adversarios[e]:
            ax.imshow(img_adversario, extent=(x+0.1, x+0.9, -y-0.1, -y-0.9), zorder=3)
        elif aliados[e]:
            ax.imshow(img_aliado, extent=(x+0.1, x+0.9, -y-0.1, -y-0.9), zorder=3)

        # Nome do estado
        ax.text(x+0.5, -y-0.5, f'E{e+1}', ha='center', va='center', fontsize=10, color='k', zorder=4)

    # Desenhar o caminho
    if caminho:
        for idx in range(len(caminho)-1):
            x1, y1 = coords[caminho[idx]]
            x2, y2 = coords[caminho[idx+1]]
            ax.arrow(x1+0.5, -y1-0.5, x2-x1, -(y2-y1),
                    head_width=0.15, head_length=0.15,
                    fc='orange', ec='orange', zorder=5)

    # Ajustar os limites para mostrar todo o grid
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-5.5, 0.5)  # CORRIGIDO: limites ajustados
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- 6. Execução Principal ---
print("Configuração inicial:")
print(f"Estado inicial: E{estado_inicial+1}")
print(f"Estado objetivo: E{estado_objetivo+1}")
print(f"Estado bola: E{estado_bola+1}")
print(f"Adversários: {[f'E{i+1}' for i in range(36) if adversarios[i]]}")

# Verificar manualmente os cálculos para E15 e E7
print("\nVerificação manual dos cálculos:")
print("Partindo de E15:")
for viz in vizinhos(estado_inicial):
    h, d, B, R = heuristica(viz, estado_objetivo, estado_bola)
    g = distancia(estado_inicial, viz)
    f = g + h
    print(f"E{viz+1}: g={g:.3f}, h={h:.3f} (d={d:.3f}, B={B:.2f}, R={R:.2f}), f={f:.3f}")

print("\nPartindo de E7:")
for viz in vizinhos(6):  # E7 é índice 6
    if viz != estado_inicial:  # Não considerar voltar para E15
        h, d, B, R = heuristica(viz, estado_objetivo, estado_bola)
        g = distancia(6, viz) + 0.5  # g(E7) = 0.5
        f = g + h
        print(f"E{viz+1}: g={g:.3f}, h={h:.3f} (d={d:.3f}, B={B:.2f}, R={R:.2f}), f={f:.3f}")

caminho, detalhes = a_star(estado_inicial, estado_objetivo, estado_bola)
if caminho is None:
    print("Nenhum caminho encontrado!")
else:
    print("\nCaminho percorrido:")
    print([f'E{e+1}' for e in caminho])
    print("\nDetalhes de cada estado no caminho:")
    print("Estado |   f(x)   |   g(x)   |   h(x)   |   d(x,ideal)   |   B(x)   |   R(x)")
    for e in caminho:
        d = detalhes[e]
        print(f"E{e+1:2}   | {d['f']:.3f} | {d['g']:.3f} | {d['h']:.3f} | {d['d']:.3f} | {d['B']:.2f} | {d['R']:.2f}")
    plot_campo(caminho, estado_bola)
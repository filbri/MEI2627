#python #ML #MEI-DAA

Chegámos ao **Matplotlib**! Se o NumPy é o motor matemático e o Pandas é a tua folha de cálculo inteligente, o Matplotlib é a tua **tela de pintura**. 

Em Machine Learning, não construímos modelos às cegas: precisas de visualizar se a função de perda (*loss*) está a diminuir, se os resíduos de uma regressão têm distribuição normal, quais as variáveis mais importantes ou como desenhar a famosa Curva ROC.


# 📊 Manual Prático & Intuitivo de Matplotlib para Data Science & ML

```python
import matplotlib.pyplot as plt
import numpy as np
```

## Módulo 1: O Fim da Confusão (Figure vs Axes)

O maior erro de quem começa é misturar `plt.plot()` com `ax.plot()`. 

Pensa no Matplotlib como uma ida a um museu:
* **`Figure` (A Moldura/Quadro):** É a janela ou a folha em branco total.
* **`Axes` (A Tela de Pintura):** É o gráfico real onde desenhas as linhas, eixos $X$ e $Y$, títulos e legendas. Uma `Figure` pode ter um ou vários `Axes`.

```
+---------------------------------------+  <- Figure
|  Título da Janela                     |
|  +-----------------+  +------------+  |
|  | Axes 1          |  | Axes 2     |  |
|  | (Gráfico 1)     |  | (Gráfico 2)|  |
|  +-----------------+  +------------+  |
+---------------------------------------+
```

### ✅ O Template Padrão Profissional (Usa sempre isto!)
```python
# Criar a Figura e os Eixos explicitamente
fig, ax = plt.subplots(figsize=(8, 4))  # largura=8, altura=4 polegadas

# Desenhar no 'ax'
ax.plot([1, 2, 3], [10, 20, 15])
ax.set_title("O meu Primeiro Gráfico Profissional")

plt.show()
```

## Módulo 2: O Gráfico Perfeito de Linha (Customização Essencial)

Em Data Science, um gráfico sem rótulos legíveis, unidades e grelha é inútil. Eis como criar um gráfico pronto para apresentação:

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(9, 4.5))

# Plot com estilos, cores, espessuras e marcadores
ax.plot(x, y1, label='Sinal A (Seno)', color='#1f77b4', linewidth=2, linestyle='-')
ax.plot(x, y2, label='Sinal B (Cosseno)', color='#ff7f0e', linewidth=2, linestyle='--')

# 1. Títulos e Legendas dos Eixos
ax.set_title("Comparação de Ondas Senoidais", fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel("Tempo (segundos)", fontsize=11)
ax.set_ylabel("Amplitude", fontsize=11)

# 2. Grelha de suporte (fundamental para leitura de valores)
ax.grid(True, linestyle=':', alpha=0.6)

# 3. Legenda
ax.legend(loc='upper right', frameon=True)

# 4. Limites dos eixos (opcional)
ax.set_ylim(-1.5, 1.5)

plt.tight_layout() # Ajusta margens automaticamente
plt.show()
```

## Módulo 3: Curvas de Treino e *Loss* (O clássico de Machine Learning)

Saber identificar *Overfitting* ou *Underfitting* requer olhar para as curvas de treino e validação ao longo das épocas (*epochs*).

```python
epocas = np.arange(1, 21)
# Perda a descer no treino, mas a subir na validação (Overfitting clássico!)
loss_treino = np.exp(-0.2 * epocas) + 0.05
loss_val = np.exp(-0.18 * epocas) + 0.03 * (epocas > 10) * (epocas - 10) * 0.1

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(epocas, loss_treino, 'b-o', markersize=4, label='Treino (Train Loss)')
ax.plot(epocas, loss_val, 'r--s', markersize=4, label='Validação (Val Loss)')

# Linha vertical a indicar onde começou o overfitting
ax.axvline(x=10, color='gray', linestyle=':', label='Ponto Ideal (Early Stopping)')

ax.set_title("Curva de Aprendizagem do Modelo", fontsize=13)
ax.set_xlabel("Épocas de Treino")
ax.set_ylabel("Função de Perda (Loss)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

## Módulo 4: Gráficos de Dispersão (*Scatter Plots*) e Regressão

Usados para identificar correlações, clusters (K-Means) ou avaliar a reta de uma Regressão Linear sobre os dados reais ($y$ vs $\hat{y}$).

```python
rng = np.random.default_rng(42)
X_horas_estudo = rng.uniform(1, 10, 50)
y_nota = 2.0 * X_horas_estudo + rng.normal(0, 2, 50)  # Relação linear com ruído

# Linha da regressão prevista (reta ideal)
reta_x = np.linspace(1, 10, 100)
reta_y = 2.0 * reta_x

fig, ax = plt.subplots(figsize=(8, 4.5))

# 1. Os pontos reais (Scatter)
# c=cor, alpha=transparência (essencial para dados sobrepostos), s=tamanho do ponto
ax.scatter(X_horas_estudo, y_nota, color='darkblue', alpha=0.7, s=60, label='Alunos Reais')

# 2. A reta do modelo
ax.plot(reta_x, reta_y, color='red', linewidth=2.5, label='Regressão Linear')

ax.set_title("Horas de Estudo vs Nota Final")
ax.set_xlabel("Horas Estudadas")
ax.set_ylabel("Nota no Exame")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

## Módulo 5: Histogramas e Análise de Resíduos (Diagnóstico de Erro)

Uma das suposições mais importantes dos modelos lineares é que os resíduos (erros $e = y - \hat{y}$) devem seguir uma **Distribuição Normal** centrada no zero.

```python
# Simulando 1000 resíduos de um modelo
residuos = rng.normal(loc=0.0, scale=1.5, size=1000)

fig, ax = plt.subplots(figsize=(8, 4.5))

# bins: número de colunas, density=True mostra densidade/probabilidade em vez de contagem
n, bins, patches = ax.hist(residuos, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)

# Linha vertical na média (deve ser aproximadamente 0)
ax.axvline(residuos.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Média: {residuos.mean():.2f}')

ax.set_title("Distribuição dos Resíduos do Modelo (Erros)")
ax.set_xlabel("Erro de Previsão")
ax.set_ylabel("Densidade")
ax.legend()
ax.grid(axis='y', alpha=0.4)

plt.show()
```

## Módulo 6: Gráficos de Barras Horizontais (*Feature Importance*)

Quando treinas um modelo baseado em árvores (Random Forest, XGBoost), queres saber quais as variáveis mais decisivas. Barras horizontais (`barh`) são o padrão da indústria porque facilitam a leitura dos nomes das variáveis.

```python
features = np.array(['Idade', 'Salário', 'Dívida', 'Score_Crédito', 'Anos_Emprego'])
importancias = np.array([0.12, 0.35, 0.28, 0.18, 0.07])

# Ordenar as features da menor para a maior relevância
indices_ordem = np.argsort(importancias)
features_ordenadas = features[indices_ordem]
importancias_ordenadas = importancias[indices_ordem]

fig, ax = plt.subplots(figsize=(8, 4.5))

# ax.barh cria barras horizontais
barras = ax.barh(features_ordenadas, importancias_ordenadas, color='#2ca02c', edgecolor='black', height=0.6)

# Adicionar o valor exato no fim de cada barra
ax.bar_label(barras, fmt='%.2f', padding=5)

ax.set_title("Importância das Variáveis (Feature Importance)", fontsize=13)
ax.set_xlabel("Importância Relativa")
ax.set_xlim(0, 0.45) # Dá espaço para o texto do rótulo
ax.grid(axis='x', linestyle=':', alpha=0.5)

plt.show()
```

## Módulo 7: Múltiplos Gráficos (*Subplots*) sem Complicação

Em Análise Exploratória (EDA), precisas frequentemente de ver gráficos lado a lado ou em grelha.

```python
# Criar grelha de 1 linha e 2 colunas
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

# axes é um array NumPy de objetos de eixos: [ax0, ax1]
# Gráfico 1 (Esquerda)
axes[0].plot(x, np.sin(x), color='purple')
axes[0].set_title("Gráfico Seno")
axes[0].grid(True)

# Gráfico 2 (Direita)
axes[1].hist(rng.normal(size=500), bins=20, color='coral', edgecolor='black')
axes[1].set_title("Distribuição Normal")
axes[1].grid(True)

# Título global para a figura inteira
fig.suptitle("Painel de Diagnóstico do Dataset", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 💡 Dica de Professor: Iterar por uma grelha 2x2 com `flatten()`
Se tiveres uma matriz de subplots `2x2`, o `axes` terá o formato `(2, 2)`. Usa `axes.flatten()` para percorreres todos com um ciclo simples:
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, ax in enumerate(axes.flatten()):
    ax.text(0.5, 0.5, f"Painel {i+1}", ha='center', va='center', fontsize=16)
    ax.set_title(f"Subplot {i+1}")
plt.tight_layout()
```

## Módulo 8: A Curva ROC (Avaliador de Classificação)

A curva ROC (*Receiver Operating Characteristic*) mede a taxa de verdadeiros positivos (TPR) contra a de falsos positivos (FPR) para diferentes limiares.

```python
# Simulando dados de uma curva ROC
fpr = np.linspace(0, 1, 100)
tpr = np.sqrt(fpr) # Curva típica de um modelo bom (AUC ~ 0.85)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. A curva do teu modelo
ax.plot(fpr, tpr, color='darkorange', lw=2.5, label='Modelo XGBoost (AUC = 0.85)')

# 2. A linha de base aleatória (chute 50/50 - Linha pontilhada diagonal)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Adivinhação Aleatória (AUC = 0.50)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
ax.set_title('Curva ROC - Desempenho do Classificador')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.show()
```

## Módulo 9: Anotações Estratégicas (`annotate` e `text`)

Para contar uma história (*Data Storytelling*), deves guiar os olhos de quem lê o relatório para o ponto crítico.

```python
x = np.arange(1, 13)
vendas = np.array([20, 22, 25, 23, 27, 26, 29, 32, 58, 35, 33, 36]) # Pico no mês 9

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, vendas, marker='o', color='teal', linewidth=2)

# Adicionar uma anotação com seta a apontar para a anomalia
ax.annotate(
    'Campanha Black Friday\n(Pico de Vendas)', 
    xy=(9, 58),             # Ponto onde a seta toca
    xytext=(5, 52),          # Onde o texto fica posicionado
    arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='lightyellow')
)

ax.set_title("Evolução Mensal de Vendas")
ax.set_xlabel("Mês")
ax.set_ylabel("Faturação (k€)")
ax.grid(True, alpha=0.3)

plt.show()
```

## Módulo 10: Guardar com Qualidade e Mudar o Estilo Global

Quando precisares de exportar o teu gráfico para um artigo, apresentação ou dashboard:

```python
# 1. Mudar o tema visual de todos os gráficos (Estilos bonitos pré-configurados)
# Podes ver os estilos disponíveis com: print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpo e moderno

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([1, 2, 3], [4, 5, 2])
ax.set_title("Gráfico com Estilo Seaborn Integrado")

# 2. Guardar a figura para ficheiro (PNG, PDF ou SVG)
# dpi=300 garante resolução nítida para impressão
# bbox_inches='tight' impede que os títulos dos eixos fiquem cortados fora da imagem!
fig.savefig("meu_grafico_alta_resolucao.png", dpi=300, bbox_inches='tight')

plt.show()
```

## 📋 Folha de Consulta Rápida de Matplotlib (Cheat Sheet)

| Se queres... | Usa este comando: |
| :--- | :--- |
| **Iniciar o gráfico profissional** | `fig, ax = plt.subplots(figsize=(L, A))` |
| **Linha contínua** | `ax.plot(x, y, label='...', color='...')` |
| **Dispersão de pontos** | `ax.scatter(x, y, alpha=0.7)` |
| **Histograma de frequências** | `ax.hist(dados, bins=30, edgecolor='black')` |
| **Barras Horizontais** | `ax.barh(categorias, valores)` |
| **Adicionar Linha de Referência** | `ax.axvline(x=valor)` ou `ax.axhline(y=valor)` |
| **Ligar/Desligar Grelha** | `ax.grid(True, linestyle=':', alpha=0.5)` |
| **Configurar Título e Rótulos** | `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()` |
| **Organizar grelha de gráficos** | `plt.tight_layout()` |
| **Exportar sem cortar bordas** | `fig.savefig('nome.png', dpi=300, bbox_inches='tight')` |


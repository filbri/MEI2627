#python #ML #MEI-DAA 

## Módulo 11: Boxplots — A Radiografia dos Outliers (`ax.boxplot`)

Antes de treinares qualquer modelo sensível a valores extremos (como Regressão Linear, SVM ou Redes Neurais), precisas de inspecionar a dispersão, a mediana e os potenciais *outliers* de cada classe ou variável.

```python
rng = np.random.default_rng(42)

# Simulando salários de três departamentos (o Dep. C tem outliers propositados)
dep_A = rng.normal(2500, 300, 50)
dep_B = rng.normal(3200, 500, 50)
dep_C = np.append(rng.normal(2800, 400, 45), [6500, 7200, 8100]) # Outliers

fig, ax = plt.subplots(figsize=(8, 4.5))

# patch_artist=True permite pintar o interior das caixas
bp = ax.boxplot([dep_A, dep_B, dep_C], 
                tick_labels=['Marketing', 'Engenharia', 'Vendas'],
                patch_artist=True,
                notch=True) # notch dá uma cintura na mediana (intervalo de confiança)

# Personalizar as cores das caixas
cores = ['#aec7e8', '#ffbb78', '#98df8a']
for patch, cor in zip(bp['boxes'], cores):
    patch.set_facecolor(cor)

ax.set_title("Distribuição Salarial por Departamento (Detetor de Outliers)")
ax.set_ylabel("Salário (€)")
ax.grid(axis='y', linestyle=':', alpha=0.6)

plt.show()
```

---

## Módulo 12: Dois Eixos no Mesmo Gráfico (`ax.twinx()` — Loss vs Accuracy)

Um dos maiores desafios ao avaliar o treino de um modelo é comparar grandezas em escalas totalmente distintas: a **Loss** (que pode ir de 0 a 5) e a **Acurácia** (que varia apenas entre 0 e 1). A solução é criar um segundo eixo $Y$ partilhando o mesmo eixo $X$.

```python
epocas = np.arange(1, 16)
loss = 2.5 * np.exp(-0.3 * epocas) + 0.1
acuracia = 0.5 + 0.45 * (1 - np.exp(-0.25 * epocas))

fig, ax1 = plt.subplots(figsize=(8, 4.5))

# 1. Primeiro Eixo Y (Loss na Esquerda - Vermelho)
cor_loss = 'tab:red'
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Loss (Função de Perda)', color=cor_loss, fontweight='bold')
linha1 = ax1.plot(epocas, loss, color=cor_loss, marker='o', label='Loss')
ax1.tick_params(axis='y', labelcolor=cor_loss)

# 2. Criar o eixo gémeo partilhando o mesmo X (Acurácia na Direita - Azul)
ax2 = ax1.twinx()
cor_acc = 'tab:blue'
ax2.set_ylabel('Acurácia (0 a 1)', color=cor_acc, fontweight='bold')
linha2 = ax2.plot(epocas, acuracia, color=cor_acc, marker='s', label='Acurácia')
ax2.tick_params(axis='y', labelcolor=cor_acc)

# Juntar as legendas de ambos os eixos numa só
linhas = linha1 + linha2
labels = [l.get_label() for l in linhas]
ax1.legend(linhas, labels, loc='center right')

ax1.set_title("Evolução Simultânea: Loss vs Acurácia")
ax1.grid(True, alpha=0.3)
plt.show()
```

---

## Módulo 13: Intervalos de Confiança e Incerteza (`ax.fill_between`)

Em validação cruzada (*K-Fold Cross-Validation*), nunca reportamos apenas a média; reportamos a média acompanhada do desvio padrão ($\mu \pm \sigma$). O `fill_between` desenha a "faixa de incerteza" do modelo.

```python
x = np.linspace(0, 10, 30)
media_score = 0.85 - 0.03 * np.sin(x)
desvio_padrao = 0.05 * np.ones_like(x) # Incerteza estimada

fig, ax = plt.subplots(figsize=(8, 4))

# 1. Linha da média
ax.plot(x, media_score, color='darkgreen', lw=2, label='Score Médio (CV)')

# 2. Área preenchida (Média - Std até Média + Std)
ax.fill_between(x, 
                media_score - desvio_padrao, 
                media_score + desvio_padrao, 
                color='green', alpha=0.2, label='Desvio Padrão ($\pm 1\sigma$)')

ax.set_title("Desempenho com Intervalo de Confiança (Cross-Validation)")
ax.set_xlabel("Complexidade do Modelo (Profundidade da Árvore)")
ax.set_ylabel("F1-Score")
ax.set_ylim(0.6, 1.0)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

plt.show()
```

---

## Módulo 14: Escalas Logarítmicas (Ajuste de Hiperparâmetros)

Quando procuras o melhor valor para parâmetros como a taxa de aprendizagem (*learning rate*) $\alpha$ ou o termo de regularização $C$ do SVM, estes variam em potências de 10 ($0.0001, 0.001, 0.01, 0.1, 1, 10$). Numa escala linear comum, os pontos menores ficariam todos esmagados contra a margem.

```python
# Taxas de aprendizagem a testar: 10^-5 até 10^0
learning_rates = np.logspace(-5, 0, 50)
# Simulando a perda que atinge um ponto ótimo em torno de 10^-2
losses = (np.log10(learning_rates) + 2)**2 + 0.5

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(learning_rates, losses, color='purple', lw=2)

# ✅ Mudar o eixo X para escala logarítmica
ax.set_xscale('log')

ax.set_title("Procura da Taxa de Aprendizagem Ideal (LR Range Test)")
ax.set_xlabel("Learning Rate (Escala Logarítmica)")
ax.set_ylabel("Perda (Loss)")
ax.grid(True, which="both", ls=":", alpha=0.5) # 'which="both"' liga a grelha maior e menor

plt.show()
```

---

## Módulo 15: Visualização de Imagens e Matrizes 2D (`ax.imshow`)

Se trabalhas com visão computacional (ex: classificar dígitos do MNIST), mapas de calor ou matrizes de confusão sem querer depender do Seaborn, o `imshow` é a ferramenta correta.

```python
# Simulando uma imagem 8x8 de um dígito (como no dataset digits do Sklearn)
imagem_digito = np.array([
    [ 0,  1,  6, 15, 12,  1,  0,  0],
    [ 0,  7, 16,  6,  6, 10,  0,  0],
    [ 0,  8, 16,  2,  0, 11,  0,  0],
    [ 0,  5, 16,  9,  9, 16,  2,  0],
    [ 0,  0,  7, 10, 15, 16,  4,  0],
    [ 0,  0,  0,  0,  2, 16,  4,  0],
    [ 0,  5, 14, 12, 13, 16,  1,  0],
    [ 0,  0,  6, 13, 10,  0,  0,  0]
])

fig, ax = plt.subplots(figsize=(5, 5))

# cmap='gray' para preto e branco; 'viridis' ou 'Blues' para cores
im = ax.imshow(imagem_digito, cmap='Blues', interpolation='nearest')

# Barra lateral de cor (Colorbar) para indicar o valor numérico dos píxeis
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensidade do Píxel')

ax.set_title("Inspeção de Imagem 8x8 (Píxeis)")
plt.show()
```

---

## Módulo 16: Controlo dos Eixos: Rotação e Ticks (`set_xticklabels`)

Um dos problemas mais comuns ao analisar datas ou nomes longos de variáveis é o texto sobrepor-se, tornando o gráfico ilegível.

```python
modelos = ['Regressão Logística', 'Random Forest Classifier', 'Gradient Boosting Machine', 'Support Vector Machine (SVM)']
precisao = [0.81, 0.89, 0.92, 0.86]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(modelos, precisao, color='teal', width=0.5)

# ✅ Rotação de 30 graus com alinhamento à direita ('ha=right') para encaixe perfeito
ax.set_xticks(range(len(modelos)))
ax.set_xticklabels(modelos, rotation=30, ha='right', fontsize=10)

ax.set_ylim(0.7, 1.0) # Focar a escala no intervalo de interesse
ax.set_ylabel("Precisão (Accuracy)")
ax.set_title("Comparativo de Modelos Preditivos")
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Módulo 17: Layouts Complexos Intuitivos com `subplot_mosaic`

Esquece o antigo e complicado `GridSpec`. Desde as versões recentes do Matplotlib, podes desenhar painéis com a sintaxe visual do **`subplot_mosaic`** — desenhando a estrutura com texto ASCII!

Imagina que queres:
* Um gráfico grande em cima (Série Temporal completa);
* Dois gráficos pequenos em baixo (Distribuição e Resíduos).

```python
# O mapa visual do teu painel:
layout = [
    ['topo', 'topo'],
    ['esquerda', 'direita']
]

fig, ax_dict = plt.subplot_mosaic(layout, figsize=(10, 6))

# Agora acedes a cada eixo pelo nome que deste!
ax_dict['topo'].plot(np.random.randn(100).cumsum(), color='navy')
ax_dict['topo'].set_title("Gráfico Principal: Tendência Temporal")

ax_dict['esquerda'].hist(np.random.randn(500), bins=20, color='crimson', alpha=0.7)
ax_dict['esquerda'].set_title("Distribuição")

ax_dict['direita'].scatter(np.random.randn(50), np.random.randn(50), color='forestgreen')
ax_dict['direita'].set_title("Dispersão dos Erros")

plt.tight_layout()
plt.show()
```

---

## Módulo 18: Estética Minimalista: Remover as Bordas (*Despine*)

Gráficos modernos em relatórios executivos evitam a "caixa preta fechada". Remover as bordas superior e direita dá um aspeto limpo e profissional ao gráfico.

```python
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot([1, 2, 3, 4], [10, 25, 20, 35], color='#333333', lw=2)

# ✅ Ocultar as linhas superior e direita (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tornar as linhas restantes mais subtis
ax.spines['left'].set_color('#888888')
ax.spines['bottom'].set_color('#888888')

ax.set_title("Design Minimalista e Limpo", fontsize=12)
plt.show()
```

---

## 📌 Tabela de Consulta Expandida (Cheat Sheet Completa)

| Desafio Visual | Comando em Matplotlib |
| :--- | :--- |
| **Detetar Outliers por Grupo** | `ax.boxplot([dados_a, dados_b], patch_artist=True)` |
| **Dois Eixos Y (Loss vs Acc)** | `ax2 = ax.twinx()` |
| **Incerteza / Desvio Padrão ($\pm\sigma$)** | `ax.fill_between(x, y_min, y_max, alpha=0.2)` |
| **Escala Logarítmica (Hiperparâmetros)** | `ax.set_xscale('log')` |
| **Exibir Matriz de Confusão / Imagem** | `ax.imshow(matriz, cmap='Blues')` |
| **Adicionar Legenda de Cores** | `fig.colorbar(im, ax=ax)` |
| **Rodar Nomes dos Eixos sem Cortar** | `ax.set_xticklabels(nomes, rotation=30, ha='right')` |
| **Remover Bordas Superiores/Direitas** | `ax.spines['top'].set_visible(False)` |
| **Construir Painéis Estilo Dashboard** | `plt.subplot_mosaic([['A', 'A'], ['B', 'C']])` |

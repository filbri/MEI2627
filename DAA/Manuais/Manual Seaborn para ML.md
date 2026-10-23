#MEI-DAA #python #ML 

Se o Matplotlib é a tela onde tu tens de pintar cada detalhe manualmente com o pincel, o Seaborn é o teu **assistente de arte especializado em Estatística e Machine Learning**. Ele foi construído diretamente em cima do Matplotlib e conversa de forma nativa com os **DataFrames do Pandas**.

Com apenas uma linha de código, o Seaborn calcula intervalos de confiança, ajusta distribuições, separa categorias por cores e formata gráficos com um padrão visual moderno e elegante.

# 🎨 Manual Prático & Intuitivo de Seaborn para Data Science & ML

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ativar o tema moderno padrão do Seaborn (fundo limpo, grelhas suaves e boa tipografia)
sns.set_theme(style="whitegrid")
```

## Módulo 1: O "Casamento Perfeito" com o Pandas e Matplotlib

A grande magia do Seaborn reside no facto de aceitar o DataFrame diretamente e mapear as colunas pelos seus nomes nos parâmetros `x`, `y` e `hue` (cor).

Além disso, podes combinar o Seaborn com a estrutura orientada a objetos do Matplotlib usando o parâmetro **`ax=ax`**:

```python
# Carregar um dataset clássico embutido no Seaborn
df_gorjetas = sns.load_dataset('tips') 

fig, ax = plt.subplots(figsize=(7, 4))

# Seaborn desenha DIRETAMENTE no objeto 'ax' do Matplotlib!
sns.scatterplot(data=df_gorjetas, x='total_bill', y='tip', hue='time', ax=ax)

# Podes continuar a usar todos os métodos do Matplotlib:
ax.set_title("Conta Total vs Gorjeta (Almoço vs Jantar)", fontsize=13)
ax.set_xlabel("Valor da Conta ($)")
ax.set_ylabel("Gorjeta ($)")

plt.show()
```

## Módulo 2: O Mapa de Calor da Correlação (`sns.heatmap`)

Este é o **gráfico nº 1** em qualquer fase de Análise Exploratória (EDA) para Machine Learning. Serve para identificar variáveis redundantes (multicolinearidade) e variáveis com forte correlação com o teu alvo.

```python
# 1. Calcular a matriz de correlação do Pandas
corr = df_gorjetas.corr(numeric_only=True)

# 💡 Truque de Mestre: Criar uma máscara triangular para esconder a metade repetida!
mascara = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(6, 5))

# annot=True coloca os números dentro das células; fmt='.2f' limita a 2 casas decimais
# cmap='coolwarm' (ou 'vlag') coloca azul no negativo e vermelho no positivo
sns.heatmap(corr, mask=mascara, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, square=True, linewidths=1, ax=ax)

ax.set_title("Matriz de Correlação Linear", fontsize=13)
plt.show()
```

## Módulo 3: Análise de Distribuições (`sns.histplot` & `sns.kdeplot`)

Antes de escolher algoritmos lineares, precisas de saber se as tuas variáveis numéricas seguem uma distribuição normal ou se são assimétricas (*skewed*).

```python
fig, ax = plt.subplots(figsize=(8, 4))

# kde=True adiciona a curva suave de densidade estimada (Kernel Density Estimation)
# hue='sex' separa a distribuição entre homens e mulheres automaticamente
sns.histplot(data=df_gorjetas, x='total_bill', kde=True, hue='sex', 
             element='step', alpha=0.3, ax=ax)

ax.set_title("Distribuição do Valor da Conta por Género")
ax.set_xlabel("Conta ($)")
plt.show()
```

## Módulo 4: A Radiografia Completa do Dataset (`sns.pairplot`)

Quando começas a explorar um problema de classificação (como o clássico *Iris* ou *Penguins*), o `pairplot` desenha todos os cruzamentos $2$ a $2$ de uma só vez.

```python
df_iris = sns.load_dataset('iris')

# corner=True corta a metade superior redundante (economiza tempo e espaço!)
# hue='species' colore os pontos pela classe alvo
grafico = sns.pairplot(df_iris, hue='species', corner=True, palette='bright')
plt.show()
```
> 💡 *Com este único comando, consegues ver de imediato quais as variáveis que separam as classes perfeitamente sem precisares de treinar um modelo complexo.*

## Módulo 5: Boxplots e Violinplots (Detetores de Outliers por Categoria)

Para comparar o comportamento de uma variável numérica entre diferentes grupos categóricos.

### A. Boxplot Clássico
```python
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df_gorjetas, x='day', y='total_bill', hue='smoker', palette='Set2', ax=ax)
ax.set_title("Gasto por Dia da Semana (Fumador vs Não Fumador)")
plt.show()
```

### B. Violinplot (A evolução do Boxplot)
O `violinplot` junta a caixa do boxplot à curva de densidade KDE, permitindo ver se a distribuição tem dois picos (bimodal):
```python
fig, ax = plt.subplots(figsize=(7, 4))
# split=True divide o violino ao meio, comparando duas categorias lado a lado!
sns.violinplot(data=df_gorjetas, x='day', y='total_bill', hue='sex', split=True, palette='Pastel1', ax=ax)
ax.set_title("Densidade de Gastos por Género em Cada Dia")
plt.show()
```

## Módulo 6: Avaliação de Classes e Desbalanceamento (`sns.countplot`)

Em problemas de Classificação (ex: Churn de clientes, Deteção de Fraude), tens de verificar se uma classe é muito mais rara do que outra.

```python
df_titanic = sns.load_dataset('titanic')

fig, ax = plt.subplots(figsize=(7, 4))

# Conta a frequência de cada categoria
# order garante que as barras ficam ordenadas da mais frequente para a menor
sns.countplot(
    data=df_titanic, 
    x='class', 
    hue='survived', 
    order=['Third', 'First', 'Second'],
    palette='muted',
    ax=ax
)

ax.set_title("Sobrevivência por Classe no Titanic")
ax.set_xlabel("Classe do Bilhete")
ax.set_ylabel("Nº de Passageiros")
plt.show()
```

## Módulo 7: Dispersão Multidimensional (`sns.scatterplot`)

No Matplotlib tínhamos de fazer loops para colorir pontos por classes. No Seaborn, podemos cruzar **4 dimensões** num só gráfico de dispersão:
1. Eixo $X$ (Numérica)
2. Eixo $Y$ (Numérica)
3. `hue` (Cor - Categórica ou Numérica)
4. `size` (Tamanho do marcador - Numérica)

```python
fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df_gorjetas,
    x='total_bill',
    y='tip',
    hue='smoker',          # Cor: Fumador ou Não
    size='size',            # Tamanho da bolha: Número de pessoas na mesa
    sizes=(20, 200),        # Intervalo de tamanhos dos marcadores
    alpha=0.7,
    palette=['red', 'blue'],
    ax=ax
)

ax.set_title("Conta vs Gorjeta por Número de Pessoas e Estatuto de Fumador")
plt.show()
```

## Módulo 8: Regressão Linear com Faixa de Incerteza (`sns.regplot` e `sns.lmplot`)

O Seaborn calcula internamente a reta de regressão por Mínimos Quadrados Ordinários (OLS) e desenha a banda de **$95\%$ de intervalo de confiança** via Bootstrap.

```python
fig, ax = plt.subplots(figsize=(7, 4.5))

# ci=95 ativa a faixa de confiança sombreada
# scatter_kws e line_kws permitem personalizar pontos e linha de forma independente
sns.regplot(
    data=df_gorjetas, 
    x='total_bill', 
    y='tip', 
    ci=95, 
    scatter_kws={'alpha': 0.5, 'color': 'gray'},
    line_kws={'color': 'red', 'linewidth': 2},
    ax=ax
)

ax.set_title("Linha de Regressão com Intervalo de Confiança (95%)")
plt.show()
```

## Módulo 9: Múltiplas Facetas em Subplots Automáticos (`sns.catplot`)

E se quiseres desenhar um gráfico separado para cada dia da semana, sem escrever ciclos `for` com Matplotlib? O `catplot` (ou `relplot`) trata de tudo criando uma grelha inteligente (*FacetGrid*):

```python
# col='day' cria uma coluna de gráfico para cada dia diferente!
# kind pode ser: 'box', 'violin', 'bar', 'count', etc.
g = sns.catplot(
    data=df_gorjetas,
    x='sex', 
    y='total_bill', 
    hue='smoker',
    col='day', 
    kind='box', 
    height=4, 
    aspect=0.8,
    palette='Set1'
)

g.set_axis_labels("", "Valor da Conta ($)")
g.set_titles("Dia: {col_name}")
plt.show()
```

## Módulo 10: Visualizar a Matriz de Confusão do Scikit-Learn

Depois de treinares o teu modelo de Machine Learning, usa o Seaborn para tornar a matriz de números confusos legível para qualquer pessoa de negócio:

```python
# Simulando a saída de confusion_matrix(y_true, y_pred) do Sklearn
matriz_confusao = np.array([
    [850, 50],   # Classe 0: Real Negativo (850 acertados, 50 falsos positivos)
    [30,  170]   # Classe 1: Real Positivo (30 falsos negativos, 170 acertados)
])

classes = ['Não Fraude', 'Fraude']

fig, ax = plt.subplots(figsize=(6, 4.5))

sns.heatmap(
    matriz_confusao, 
    annot=True, 
    fmt='d',           # 'd' formata como inteiro em vez de notação científica
    cmap='Blues', 
    xticklabels=classes, 
    yticklabels=classes,
    cbar=False,
    ax=ax
)

ax.set_xlabel('Previsão do Modelo', fontweight='bold')
ax.set_ylabel('Classe Real', fontweight='bold')
ax.set_title('Matriz de Confusão do Classificador', fontsize=13)

plt.show()
```

## 📋 Folha de Consulta Rápida de Seaborn (Cheat Sheet)

| Objetivo em Data Science / ML | Função Seaborn |
| :--- | :--- |
| **Matriz de Correlação das Features** | `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')` |
| **Distribuição de uma Feature com KDE** | `sns.histplot(df, x='col', kde=True, hue='classe')` |
| **Explorar todos os pares de Features** | `sns.pairplot(df, hue='alvo', corner=True)` |
| **Detetar Outliers por Grupo** | `sns.boxplot(df, x='cat', y='num', hue='subcat')` |
| **Verificar Desbalanceamento de Classes** | `sns.countplot(df, x='target')` |
| **Relação de 2 Variáveis + Linha OLS** | `sns.regplot(df, x='x', y='y')` |
| **Dispersão com Cor e Tamanho** | `sns.scatterplot(df, x='x', y='y', hue='c', size='s')` |
| **Dividir gráficos por categorias (Facet)** | `sns.catplot(df, x='x', y='y', col='categoria')` |
| **Matriz de Confusão Bonita** | `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')` |

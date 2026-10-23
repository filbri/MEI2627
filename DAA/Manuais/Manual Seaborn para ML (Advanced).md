#MEI-DAA #python #ML 

## Módulo 11: Múltiplas Experiências e Tendências (`sns.lineplot`)

No Matplotlib, para desenhares uma linha com a média e uma sombra de desvio padrão entre 5 testes diferentes, tinhas de fazer contas à mão com `np.mean()`, `np.std()` e `ax.fill_between()`. 

O `sns.lineplot` faz tudo isso **automaticamente**! Se tiveres várias observações para o mesmo valor de $X$ (por exemplo, 5 sementes aleatórias de um modelo de ML), ele calcula a média e a banda de $95\%$ de confiança por *bootstrap*.

```python
# Simulando 5 execuções (seeds) de treino ao longo de 10 épocas
np.random.seed(42)
dados_treino = []
for seed in range(5):
    for epoca in range(1, 11):
        acuracia = 0.6 + 0.35 * (1 - np.exp(-0.4 * epoca)) + np.random.normal(0, 0.03)
        dados_treino.append({'Epoca': epoca, 'Acuracia': acuracia, 'Modelo': 'Rede Neural'})

df_exp = pd.DataFrame(dados_treino)

fig, ax = plt.subplots(figsize=(7, 4))

# errorbar='ci' (intervalo de confiança 95%) ou errorbar='sd' (desvio padrão)
sns.lineplot(data=df_exp, x='Epoca', y='Acuracia', hue='Modelo', errorbar='sd', ax=ax)

ax.set_title("Desempenho Médio ± Desvio Padrão (5 Random Seeds)")
plt.show()
```

---

## Módulo 12: Barras Estatísticas com Estimadores (`sns.barplot`)

Ao contrário do Matplotlib (onde o gráfico de barras apenas desenha os valores que passas), o `sns.barplot` **calcula estatísticas**. Por padrão, ele calcula a **média** e desenha a barra de erro da incerteza.

Podes mudar o cálculo estatístico passando o parâmetro `estimator` (ex: mediana em dados com muitos *outliers*):

```python
df_tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(7, 4))

# Compara a MEDIANA da conta por dia (evita distorção de contas gigantes)
sns.barplot(
    data=df_tips, 
    x='day', 
    y='total_bill', 
    estimator=np.median, 
    errorbar=('ci', 95), # Intervalo de confiança da mediana
    palette='Blues_d',
    ax=ax
)

ax.set_title("Mediana da Conta por Dia (com 95% de Confiança)")
ax.set_ylabel("Mediana do Gasto ($)")
plt.show()
```

---

## Módulo 13: O Gráfico Conjunto Bivariado + Marginais (`sns.jointplot`)

Em Machine Learning, quando queres cruzar duas variáveis contínuas cruciais, não queres só o gráfico de dispersão: queres ver também como cada variável se distribui isoladamente nas margens.

O `sns.jointplot` cria um gráfico central acompanhado de dois histogramas no topo e à direita:

```python
# kind pode ser 'scatter', 'kde', 'hist' ou 'reg' (com linha de regressão!)
g = sns.jointplot(
    data=df_tips, 
    x='total_bill', 
    y='tip', 
    hue='sex',
    kind='scatter',
    marginal_kws=dict(fill=True), # Preenche os histogramas laterais
    palette=['#3470a3', '#e06d53'],
    height=6
)

g.figure.suptitle("Relação Conta vs Gorjeta + Distribuições Marginais", y=1.02)
plt.show()
```

---

## Módulo 14: Ver os Dados Reais sem Ilusões (`stripplot` sobreposto ao `boxplot`)

Um problema comum do `boxplot` é que uma caixa pode representar 10 clientes ou 10.000 clientes (ele esconde o volume real de dados). 

O truque profissional é sobrepor um **`stripplot`** com pontos translúcidos e dispersos (*jitter*) por cima do *boxplot*:

```python
fig, ax = plt.subplots(figsize=(8, 4.5))

# 1. Desenhar a estrutura estatística (caixa cinzenta suave)
sns.boxplot(data=df_tips, x='day', y='total_bill', color='#eeeeee', showfliers=False, ax=ax)

# 2. Desenhar os pontos reais por cima com jitter (tremor horizontal para não sobrepor)
sns.stripplot(data=df_tips, x='day', y='total_bill', hue='time', 
              jitter=0.2, alpha=0.7, size=6, palette='Set1', ax=ax)

ax.set_title("Boxplot Enriquecido com Pontos Reais")
plt.show()
```

---

## Módulo 15: O Clustermap (Agrupamento Hierárquico Automático)

Esta é uma das funções mais impressionantes do ecossistema Python. O `sns.clustermap` não calcula apenas a matriz de correlação: ele roda nos bastidores um algoritmo de **Machine Learning Não Supervisionado (Hierarchical Clustering)** e reorganiza as linhas e colunas para agrupar as variáveis mais parecidas entre si.

```python
# Correlação entre variáveis numéricas
corr = df_tips.corr(numeric_only=True)

# O clustermap cria a sua própria figura com dendrogramas
g = sns.clustermap(
    corr, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, vmax=1,
    figsize=(6, 6),
    dendrogram_ratio=(0.15, 0.15), # Tamanho das árvores de decisão laterais
    cbar_pos=(1.02, 0.2, 0.03, 0.6) # Posição da barra de escala
)

g.figure.suptitle("Clustermap: Variáveis Agrupadas por Similaridade", y=1.05)
plt.show()
```
> 💡 *Se duas variáveis tiverem um ramo do dendrograma muito curto a uni-las, significa que carregam praticamente a mesma informação (redutíveis por PCA ou seleção de features).*

---

## Módulo 16: A Distribuição Acumulada Perfeita (`sns.ecdfplot`)

Histogramas dependem do tamanho das barras (*bins*). Se mudares o número de barras, a curva muda. 

A **Função de Distribuição Cumulativa Empírica (ECDF)** é a melhor alternativa moderna para responder a perguntas de negócio do tipo: *"Que percentagem dos clientes gasta menos de 20 dólares?"*

```python
fig, ax = plt.subplots(figsize=(7, 4))

# Plota a percentagem acumulada de 0 a 100% (0.0 a 1.0)
sns.ecdfplot(data=df_tips, x='total_bill', hue='time', ax=ax)

# Linha de referência nos 80% dos clientes
ax.axhline(0.80, color='gray', linestyle='--', alpha=0.7)
ax.set_title("Percentagem Acumulada de Gastos (Almoço vs Jantar)")
ax.set_xlabel("Conta ($)")
ax.set_ylabel("Proporção de Clientes (0 a 1)")

plt.show()
```

---

## Módulo 17: Paletas de Cores Profissionais e Acessíveis

Em apresentações de Machine Learning, a escolha incorreta das cores confunde o público. O Seaborn categoriza as paletas em 3 famílias lógicas:

1. **Qualitativas (Classes distintas sem ordem):**  
   * Exemplos: `'Set2'`, `'tab10'`, `'bright'`  
   * *Uso:* Classes alvo ($y \in \{\text{Gato, Cão, Pássaro}\}$).
2. **Sequenciais (Valores numéricos de baixo para alto):**  
   * Exemplos: `'Blues'`, `'viridis'`, `'rocket'`  
   * *Uso:* Idade, Renda, Importância das Features.
3. **Divergentes (Valores com um ponto central neutro, ex: zero):**  
   * Exemplos: `'coolwarm'`, `'vlag'`, `'RdBu'`  
   * *Uso:* Correlações ($-1$ a $+1$), Lucro vs Prejuízo.

```python
# 💡 Dica: Podes definir a paleta global do teu notebook logo no início:
sns.set_palette("Set2")
```

---

## 📌 Cheat Sheet de Seaborn Completa & Definitiva

| Necessidade Analítica                            | A Melhor Função de Seaborn                                 |
| :----------------------------------------------- | :--------------------------------------------------------- |
| **Média e Incerteza de Múltiplos Treinos**       | `sns.lineplot(df, x='epoca', y='acc', errorbar='sd')`      |
| **Comparar Estatística por Grupo (com IC)**      | `sns.barplot(df, x='grupo', y='num', estimator=np.median)` |
| **Correlação com Agrupamento Automático**        | `sns.clustermap(df.corr(), annot=True, cmap='coolwarm')`   |
| **Dois Eixos Numéricos + Histogramas nos Topos** | `sns.jointplot(df, x='x', y='y', kind='reg')`              |
| **Evitar Ilusões de Amostras Pequenas**          | `sns.boxplot(...)` + `sns.stripplot(..., jitter=0.2)`      |
| **Calcular Percentis Reais sem Barras**          | `sns.ecdfplot(df, x='coluna', hue='classe')`               |
| **Verificar Relações Multidimensionais**         | `sns.scatterplot(df, x='x', y='y', hue='c', size='s')`     |
| **Matriz de Confusão / Heatmap Simples**         | `sns.heatmap(matriz, annot=True, fmt='d')`                 |
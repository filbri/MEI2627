# Teórica
## 1. Anatomia e Conceito Fundamental

Uma **Árvore de Decisão** é um modelo preditivo supervisionado que particiona o espaço de dados através de decisões hierárquicas e ortogonais:
* **Nó Raiz (*Root Node*):** Ponto de partida que contém todos os dados e testa o atributo mais discriminante.
* **Nós Internos / Ramos:** Testes sobre atributos discretos ou contínuos (ex.: $\text{Idade} > 40$, $\text{Dosagem} < 14.5$).
* **Nós Folha (*Leaf Nodes*):** Terminais da árvore que fornecem a previsão final:
  * Em **Classificação:** Rótulo da classe maioritária (ou probabilidades).
  * Em **Regressão:** Valor numérico contínuo (a **média aritmética** das amostras que caíram nessa folha).

---

## 2. Árvores de Classificação (*Classification Trees*)

O objetivo consiste em criar nós filhos sucessivamente mais **puros** (onde predomina uma única classe).

### A. Medidas de Impureza
* **Impureza de Gini (*Gini Impurity*):** Mede a probabilidade de uma amostra ser classificada incorretamente se for rotulada aleatoriamente de acordo com a distribuição das classes no nó.
  $$G = 1 - \sum_{i=1}^{C} (P_i)^2$$
  *(Para duas classes: $G = 1 - P_{\text{sim}}^2 - P_{\text{não}}^2$. Se o nó for $100\%$ puro, $G = 0$; se for divisão $50/50$, $G = 0.5$).*
* **Entropia (*Entropy*):** Medida da teoria da informação que avalia o nível de desordem.
* **Ganho de Informação (*Information Gain*):** A redução líquida de impureza obtida após a divisão de um nó.

---

### B. Cálculo Passo-a-Passo da Impureza de Gini

Considere a decisão de diagnosticar **Doença Cardíaca** avaliando o atributo `Dor no Peito` (*Chest Pain*):

```
                       [ Dor no Peito? ]
                        /             \
                    Sim                 Não
                   (144)               (159)
             [Sim: 105, Não: 39]  [Sim: 34, Não: 125]
```

1. **Gini do Ramo Esquerdo ($N=144$):**
   $$G_{\text{esq}} = 1 - \left(\frac{105}{144}\right)^2 - \left(\frac{39}{144}\right)^2 = 1 - (0.729)^2 - (0.271)^2 \approx \mathbf{0.395}$$

2. **Gini do Ramo Direito ($N=159$):**
   $$G_{\text{dir}} = 1 - \left(\frac{34}{159}\right)^2 - \left(\frac{125}{159}\right)^2 = 1 - (0.214)^2 - (0.786)^2 \approx \mathbf{0.336}$$

3. **Impureza de Gini Total Ponderada ($TG$):**
   $$TG = \frac{N_{\text{esq}}}{N_{\text{total}}} G_{\text{esq}} + \frac{N_{\text{dir}}}{N_{\text{total}}} G_{\text{dir}}$$
   $$TG = 0.395 \times \left(\frac{144}{303}\right) + 0.336 \times \left(\frac{159}{303}\right) = \mathbf{0.364}$$

4. **Regra de Decisão do Algoritmo:**
   Compara-se o $TG$ de todos os atributos candidatos:
   * $G_{\text{ChestPain}} = 0.364$
   * $G_{\text{BloodCirculation}} = \mathbf{0.360}$ $\longleftarrow$ **Menor impureza: Escolhido como Nó Raiz**
   * $G_{\text{BlockedArteries}} = 0.381$

---

### C. Divisão de Atributos Contínuos
Para variáveis numéricas (ex.: `Peso` $= [155, 180, 190, 220, 225]$):
1. Ordenam-se os dados por ordem crescente.
2. Calculam-se os pontos médios entre valores adjacentes ($167.5, 185, 205, 222.5$).
3. Trata-se cada ponto médio como um limiar binário candidato (ex.: $\text{Peso} < 205$).
4. Escolhe-se o limiar que produz a **menor impureza de Gini ponderada**.

---

## 3. Árvores de Regressão (*Regression Trees*)

Enquanto uma regressão linear projeta uma reta contínua global, a árvore de regressão divide o espaço em degraus constantes.

```
   Regressão Linear (Reta Global)        Árvore de Regressão (Patamares/Médias)
            y ^                                  y ^
              |    /                               |       ┌──────── (Média 3)
              |   /                                |  ┌────┘ (Média 2)
              |  /                                 |──┘ (Média 1)
              +-------> x                          +-------> x
```

### A. Critério de Divisão: Soma dos Quadrados dos Resíduos (SSR)
O algoritmo procura o limiar de divisão que minimiza a soma do erro quadrático face à média local de cada partição:
$$\text{SSR} = \sum_{j \in \text{Esq}} (y_j - \bar{y}_{\text{esq}})^2 + \sum_{k \in \text{Dir}} (y_k - \bar{y}_{\text{dir}})^2$$

### B. Procedimento de Construção
1. Para cada limiar candidato num atributo, calcula-se a média das amostras à esquerda ($\bar{y}_{\text{esq}}$) e à direita ($\bar{y}_{\text{dir}}$).
2. Calcula-se a $\text{SSR}$ resultante dessa divisão.
3. Repete-se para todos os atributos e escolhe-se o par `(atributo, limiar)` com o **menor $\text{SSR}$ global**.
4. **Previsão:** Uma nova instância percorre a árvore até uma folha, assumindo como previsão o **valor médio** de treino contido nessa folha.

---

## 4. Combate ao Sobreajuste: Estratégias de Poda (*Pruning*)

Uma árvore que cresça até à pureza absoluta ($G=0$ ou $\text{SSR}=0$) memoriza ruído e sofre de **sobreajuste (*overfitting*) extremo**.

```
                           CONTROLO DE COMPLEXIDADE
                      ________________|________________
                     |                                 |
           Pré-Poda (Early Stopping)         Pós-Poda (Cost-Complexity)
           - Trava o crescimento antes       - Cresce a 100% e depois poda
           - max_depth, min_samples_split    - ccp_alpha via Cross-Validation
```

### 1. Pré-Poda (*Pre-Pruning / Early Stopping*)
Interrompe a expansão da árvore durante o treino caso determinados limiares sejam atingidos:
* `max_depth`: Limita a profundidade máxima permitida.
* `min_samples_split`: Número mínimo de instâncias exigidas num nó para permitir uma nova ramificação.
* `min_samples_leaf`: Número mínimo de instâncias que uma folha tem de conter obrigatoriamente.

### 2. Pós-Poda (*Post-Pruning*): *Cost Complexity Pruning*
Permite que a árvore cresça na totalidade e elimina de seguida os ramos que trazem ganhos residuais face à sua complexidade. Avalia o compromisso através do índice de custo-complexidade $R_\alpha(T)$:
$$R_\alpha(T) = R(T) + \alpha |T|$$

* $R(T)$: Erro total de treino nas folhas (impureza total ou $\text{SSR}$).
* $|T|$: Número de folhas da árvore (medida de complexidade).
* $\alpha$ (*Complexity Parameter / `ccp_alpha`*): Penalização por cada folha adicional.
  * Se $\alpha = 0$: A árvore original completa é mantida.
  * À medida que $\alpha$ aumenta: Árvores mais compactas e podadas são favorecidas.
* **Implementação Scikit-Learn:** O método `clf.cost_complexity_pruning_path(X_train, y_train)` devolve os valores ótimos de $\alpha$, permitindo selecionar o melhor via `GridSearchCV` com validação cruzada.

---

## 5. Família de Algoritmos Clássicos de Árvores

| Algoritmo | Autor / Origem | Tipo de Divisão | Métrica de Impureza | Suporte de Atributos | Poda Integrada |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ID3** | Ross Quinlan (1986) | Multi-ramo | Entropia / Ganho de Info. | Apenas Categóricos | Não |
| **C4.5** | Ross Quinlan (1993) | Multi-ramo | Taxa de Ganho (*Gain Ratio*) | Discretos e Contínuos (lida com nulos) | Sim (Pós-poda) |
| **J48** | Open-source (WEKA) | Multi-ramo | Taxa de Ganho (*Gain Ratio*) | Implementação Java do C4.5 | Sim |
| **CART** | Leo Breiman et al. (1984) | **Estritamente Binária** | **Gini** (Classif.) / **SSR** (Regressão) | Discretos e Contínuos (Base do Scikit-Learn) | Sim (*Cost-Complexity*) |
| **CHAID** | Vários autores | Multi-nível | Teste Qui-Quadrado ($\chi^2$) | Categóricos (muito usado em Marketing) | Paragem estatística |

---

## 6. Balanço Crítico: Vantagens vs. Fragilidades

### Pontos Fortes (*Strengths*)
* **Interpretabilidade Total (*White-Box*):** As decisões mapeiam-se em regras lógicas `SE ... ENTÃO ...`, facilmente explicáveis a equipas e auditores.
* **Pré-Processamento Mínimo:** **Não exige normalização nem padronização/escalonamento** dos dados (a ordem dos valores não altera o ponto de corte do limiar).
* **Robustez Estrutural:** Não é diretamente sensível a *outliers* univariados em variáveis explicativas.

### Fragilidades (*Weaknesses*)
* **Alta Variância / Instabilidade Estrutural:** Uma pequena alteração no conjunto de treino pode alterar o nó raiz e modificar por completo toda a árvore.
* **Fronteiras Ortogonais Rígidas:** Têm dificuldade em modelar diagonais e interações suaves entre variáveis sem construir sub-árvores replicadas.
* **Custo Computacional de Treino:** A procura gulosa (*greedy search*) exaustiva por limiares ótimos em variáveis contínuas com muitas observações pode ser exigente em termos de memória e processamento.

# Prática

# Manual Prático de Regressão Linear e Logística
**Casos Reais: USA Housing & Titanic — DAA / Universidade do Minho (Parte IV)**

---

## 0. Enquadramento no Fluxo de Machine Learning

Ambos os problemas inserem-se no paradigma de **Aprendizagem Supervisionada** (dados rotulados onde o objetivo é mapear $y = f(X)$):

$$\text{Ingestão} \longrightarrow \text{EDA \& Limpeza} \longrightarrow \text{Segregação (Treino / Teste)} \longrightarrow \text{Treino} \longrightarrow \text{Validação \& Métricas}$$

---

## MÓDULO 1: Regressão Linear Aplicada (Caso USA Housing)

**Problema:** Prever a variável contínua `Price` (preço de venda de imóveis) com base em variáveis socioeconómicas da zona.

```
[USA_Housing.csv] ──> [EDA / Heatmap / Pairplot] ──> [LinearRegression] ──> [Validação dos Resíduos]
```

### 1. Exploração e Preparação dos Dados
* **Inspeção Inicial:** O dataset possui 5.000 registos numéricos sem nulos.
* **Seleção de Variáveis:** O atributo `Address` é removido por ser texto não estruturado na forma bruta.
* **Segregação:** Divisão em **60% Treino** e **40% Teste** (`test_size=0.4, random_state=101`).

```python
import pandas as pd
from sklearn.model_selection import train_test_split

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```

---

### 2. Treino e Interpretação dos Coeficientes
O modelo estima uma equação linear da forma:
$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p$$

A interpretação é feita sob a premissa ***ceteris paribus*** (mantendo todas as outras variáveis fixas):

| Variável | Coeficiente ($\beta_j$) | Efeito no Preço ($) |
| :--- | :---: | :--- |
| **`Avg. Area Income`** | $+21.53$ | $+1\$$ na renda média $\implies \mathbf{+\$21.53}$ |
| **`Avg. Area House Age`** | $+164,883.28$ | $+1$ ano de idade média $\implies \mathbf{+\$164,883.28}$ |
| **`Avg. Area Number of Rooms`** | $+122,368.68$ | $+1$ divisão na habitação $\implies \mathbf{+\$122,368.68}$ |
| **`Avg. Area Number of Bedrooms`**| $+2,233.80$ | $+1$ quarto na habitação $\implies \mathbf{+\$2,233.80}$ |
| **`Area Population`** | $+15.15$ | $+1$ residente na cidade $\implies \mathbf{+\$15.15}$ |
| **_Intercepto ($\beta_0$)_** | $-2,640,159.80$ | Preço teórico de base |

---

### 3. Diagnóstico dos Resíduos e Métricas de Erro

Para validar os pressupostos da regressão linear, analisa-se o erro de previsão: $\text{Resíduo} = y_{\text{test}} - \hat{y}$.

1. **Dispersão ($y_{\text{real}}$ vs. $\hat{y}$):** Os pontos devem distribuir-se harmoniosamente sobre uma diagonal perfeita ($y=x$).
2. **Histograma dos Resíduos:** Deve ter a forma de uma **distribuição Gaussiana centrada em zero**. Desvios graves indicam que o modelo ignora padrões não-lineares.

```python
from sklearn import metrics
import numpy as np

predictions = lm.predict(X_test)

mae = metrics.mean_absolute_error(y_test, predictions)  # 82,288$  (Erro médio linear)
mse = metrics.mean_squared_error(y_test, predictions)   # 1.04e10$² (Penaliza erros grandes)
rmse = np.sqrt(mse)                                     # 102,278$ (Erro na unidade original)
```

---

## MÓDULO 2: Regularização LASSO ($L_1$) na Prática

A regularização adiciona uma penalização à função de perda para combater o *overfitting* e conter coeficientes inflacionados.

### Experiência com Ruído Sintético
Foram criadas e inseridas no dataset duas colunas puramente aleatórias (`randomCol1` e `randomCol2`).

```python
from sklearn.linear_model import Lasso

# Regressão Linear Padrão
lin_reg = LinearRegression().fit(X, y)

# Regressão com Regularização L1 (alpha = 2000)
lasso = Lasso(alpha=2000, random_state=2025).fit(X, y)
```

### Comparação do Impacto nos Coeficientes:
* **`randomCol1`:** O modelo linear padrão atribuiu-lhe um peso espúrio de **$220.94$**. O LASSO reduziu o coeficiente **exatamente a $0.00$** (seleção automática da variável).
* **`randomCol2`:** O LASSO encolheu o valor de $134.50$ para $125.11$, mas não o anulou.
* **Conclusão:** O LASSO tem capacidade inerente de **seleção de atributos** ao zerar coeficientes inúteis. Contudo, perante variáveis com magnitudes elevadas sem escalonamento prévio (*feature scaling*), pode não regularizar todo o ruído de forma ótima.

---

## MÓDULO 3: Regressão Logística Aplicada (Caso Titanic)

**Problema:** Prever a variável binária discreta `Survived` ($0$ = Não sobreviveu, $1$ = Sobreviveu).

---

### 1. Tratamento Estratégico de Dados em Falta

A análise via mapa de calor (`sns.heatmap(train.isnull())`) exigiu duas abordagens distintas:

```
                  ┌─ Cabin ( >70% nulos ) ──────> ELIMINAR a coluna (ou criar flag binária)
Missing Values ──┤
                  └─ Age ( ~20% nulos ) ────────> IMPUTAÇÃO CONDICIONAL
```

* **Imputação Condicional de `Age`:** Em vez da média global, usou-se a **mediana por classe (`Pclass`)**, visto que passageiros de classes superiores eram tendencialmente mais velhos:
  ```python
  train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('median'))
  train.drop('Cabin', axis=1, inplace=True)
  train.dropna(inplace=True)  # Remove as 2 linhas residuais de 'Embarked'
  ```

---

### 2. Codificação Categórica (*Dummy Variables*)

Modelos lineares não processam variáveis textuais. Converte-se `Sex` e `Embarked` em variáveis binárias, utilizando `drop_first=True` para prevenir a **multicolinearidade perfeita** (*Dummy Variable Trap*):

```python
sex = pd.get_dummies(train['Sex'], drop_first=True)         # Gera coluna 'male' (0=Mulher, 1=Homem)
embark = pd.get_dummies(train['Embarked'], drop_first=True) # Gera colunas 'Q' e 'S' (referência: 'C')

# Descarta atributos não numéricos irrelevantes
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)
```

---

### 3. Guia de Otimizadores (*Solvers*) no Scikit-Learn

A classe `LogisticRegression` suporta vários algoritmos para resolver o problema de otimização numérica:

| *Solver* | Penalizações Suportadas | Escala de Dados Recomendada | Multiclasse Nativo (*Multinomial*) |
| :--- | :---: | :--- | :---: |
| **`liblinear`** | **$L_1$, $L_2$** | **Datasets Pequenos** | Não (Usa *One-vs-Rest*) |
| **`lbfgs`** *(default)* | $L_2$, None | Médios / Grandes | Sim |
| **`newton-cg`** | $L_2$, None | Médios / Grandes | Sim |
| **`sag`** | $L_2$, None | **Muito Grandes** | Sim |
| **`saga`** | **ElasticNet, $L_1$, $L_2$, None** | **Muito Grandes** | Sim |

* **Tempo de Execução (no Titanic):** O `liblinear` e o `newton-cg` revelaram tempos de convergência quase instantâneos ($\approx 0.01$ s), sendo o `liblinear` a escolha mais ágil para problemas tabulares desta escala.

---

### 4. Avaliação e Métricas de Classificação

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Relatório Completo por Classe
print(classification_report(y_test, predictions))

# Matriz de Confusão Gráfica
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
```

#### Leitura dos Resultados:
* **Classe 0 (Não Sobreviveu):**
  * *Recall* $\approx 0.91$ / *Precision* $\approx 0.82 \implies$ O modelo identifica muito bem quem faleceu (poucos falsos positivos para a morte).
* **Classe 1 (Sobreviveu):**
  * *Precision* $\approx 0.84$ / *Recall* $\approx 0.68 \implies$ Quando o modelo afirma que um passageiro sobreviveu, acerta em $84\%$ das vezes. Contudo, falha em detetar cerca de $32\%$ dos sobreviventes reais (*Falsos Negativos*).
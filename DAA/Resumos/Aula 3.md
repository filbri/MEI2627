# Teórica

## 1. Fundamentos dos Modelos Lineares

Os modelos lineares assentam na simplicidade analítica e eficiência computacional. Satisfazem as propriedades de linearidade:
$$f(x + y) = f(x) + f(y) \quad \text{e} \quad f(\alpha x) = \alpha f(x)$$

* **Aplicações:** Regressão (previsão de valores contínuos) ou Classificação (definição de um hiperplano separador entre classes).
* **Limitação Teórica Central:** **Não resolvem problemas não-lineares** de forma direta (ex.: o problema lógico do **XOR** ou curvas de dados de grau superior sem transformação prévia de atributos).

---

## 2. Regressão Linear (Previsão Contínua)

O objetivo consiste em ajustar uma reta (ou hiperplano) a uma nuvem de pontos observada de modo a prever saídas contínuas $y$.

### A. Formulação Matemática
* **Regressão Linear Simples ($n = 1$ atributo):**
  $$\hat{y}^{(i)} = \theta_0 + \theta_1 x_1^{(i)}$$
  * $\theta_0$ (*Intercept*): Ponto de corte no eixo $Y$.
  * $\theta_1$ (*Slope* / Declive): Taxa de variação $\frac{\Delta y}{\Delta x}$.

* **Regressão Linear Múltipla ($n \ge 2$ atributos):**
  $$\hat{y}^{(i)} = h_\theta(x^{(i)}) = \theta_0 + \sum_{j=1}^{n} \theta_j x_j^{(i)} = \boldsymbol{\theta}^T \mathbf{x}^{(i)}$$
  * Cada coeficiente $\theta_j$ quantifica a contribuição marginal independente do atributo $x_j$ para a variável resposta $y$.

### B. Função de Custo / Perda: Erro Quadrático Médio ($J_\theta$)
Para encontrar a melhor reta, minimiza-se o desvio quadrático entre as estimativas e os valores reais (MSE, com o fator auxiliar $\frac{1}{2}$ para simplificar a derivação):
$$J(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$
*(onde $m$ é o número de instâncias/exemplos no dataset).*

---

## 3. Estimação de Parâmetros: Mínimos Quadrados vs. Gradiente Descendente

Para minimizar $J(\boldsymbol{\theta})$, existem duas abordagens fundamentais:

```
                            ESTIMAÇÃO DE PARÂMETROS
                       ________________|________________
                      |                                 |
           Método Analítico                  Método Iterativo
       (Mínimos Quadrados / Normal Eq.)    (Gradiente Descendente)
       - Solução matemática exata          - Passos sucessivos guiados por α
       - Sem hiperparâmetros               - Escala para grandes volumes
```

### A. Método Analítico: Equação Normal (Mínimos Quadrados)
Resolve algebricamente o sistema onde o vetor gradiente se anula ($\nabla J(\boldsymbol{\theta}) = 0$):
$$\boldsymbol{\theta} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$
* $\mathbf{X}$ é a matriz de dados com uma primeira coluna de $1$'s (para acomodar o termo independente $\theta_0$).
* **Vantagens:** Garante a solução ótima global sem necessidade de iterações nem de afinação de taxa de aprendizagem ($\alpha$).
* **Desvantagens:** Exige a inversão da matriz $\mathbf{X}^T \mathbf{X}$ (complexidade aproximada $\mathcal{O}(n^3)$). Torna-se **computacionalmente inviável para datasets com elevado número de atributos** ($n > 10^4$).

### B. Método Iterativo: Gradiente Descendente (*Gradient Descent - GD*)
Atualiza simultaneamente todos os parâmetros $\theta_j$ na direção oposta à derivada parcial da função de custo:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta}) \implies \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

* **Taxa de Aprendizagem ($\alpha$ / *Learning Rate*):**
  * $\alpha$ **muito pequeno:** Convergência garantida, mas excessivamente lenta.
  * $\alpha$ **demasiado grande:** Otimização oscilatória, ultrapassa o mínimo (*overshoot*) ou diverge.

### C. Variantes do Gradiente Descendente

| Variante | Amostra usada por iteração | Desempenho com muitas Instâncias ($m$) | Desempenho com muitos Atributos ($n$) |
| :--- | :--- | :---: | :---: |
| **Mínimos Quadrados** | Analítico direto (matriz global) | Rápido | **Lento / Inviável** |
| **Batch GD** | Todas as $m$ instâncias | **Lento** | Rápido |
| **Stochastic GD (SGD)** | 1 única instância aleatória | **Rápido** | Rápido |
| **Mini-Batch GD** | Pequeno lote (*batch*, ex.: 32 a 256 instâncias) | **Rápido (Melhor compromisso)** | Rápido |

---

## 4. Regressão Logística (Classificação Binária e Multiclasse)

Embora tenha o termo "regressão", é um algoritmo de **classificação**. Modela a probabilidade de uma instância pertencer à classe $1$.

### A. A Função Sigmoide (Logística)
Mapeia qualquer valor real no intervalo aberto $]0, 1[$:
$$p = h_\theta(x) = \sigma(\boldsymbol{\theta}^T \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^T \mathbf{x}}}$$

* **Regra de Decisão:** Define-se habitualmente o limiar (*threshold*) em $0.5$:
  * Se $p \ge 0.5 \implies \hat{y} = 1$
  * Se $p < 0.5 \implies \hat{y} = 0$

### B. Função de Custo: *Log Loss* (Entropia Cruzada Binária)
O erro quadrático (MSE) tornaria a superfície de custo da regressão logística não-convexa (cheia de mínimos locais). Usa-se a função logarítmica convexa:
$$\text{Cost}(h_\theta(x), y) = \begin{cases} -\log\left(h_\theta(x)\right) & \text{se } y = 1 \\ -\log\left(1 - h_\theta(x)\right) & \text{se } y = 0 \end{cases}$$
* Se $y = 1$ e a previsão $p \to 1 \implies \text{Erro} = 0$. Se $p \to 0$, a penalização tende para $\infty$.
* Se $y = 0$ e a previsão $p \to 0 \implies \text{Erro} = 0$. Se $p \to 1$, a penalização tende para $\infty$.

### C. Extensão para Múltiplas Classes: Estratégia *One-vs-Rest* (OvR)
1. Para $K$ classes distintas, treinam-se **$K$ classificadores binários independentes**.
2. O classificador $k$ aprende a distinguir a classe $k$ (positiva) de todas as outras agrupadas (negativas).
3. Na inferência de um novo dado, avaliam-se os $K$ modelos e atribui-se a **classe com maior probabilidade prevista**: $\hat{y} = \arg\max_k (p_k)$.

---

## 5. Controlo da Complexidade e Sobreajuste (*Overfitting*)

```
   [ Underfitting ]             [ Modelo Adequado ]               [ Overfitting ]
Linear (Grau 1 rígido)      Polinomial com Regularização        Complexidade Excessiva
 Elevado Viés (Bias)              Boa Generalização            Alta Variância (Variance)
(Não capta o padrão)                                          (Memoriza o ruído/amostra)
```

### Regularização de Modelos Lineares
A regularização penaliza coeficientes de grande magnitude na função objetivo para evitar fronteiras de decisão excessivamente complexas:

$$\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \text{Penalização} \right]$$

* **Lasso Regression (Regularização L1):**
  $$\text{Penalização} = \lambda \sum_{j=1}^{p} |\beta_j|$$
  * **Propriedade fundamental:** Força coeficientes de variáveis irrelevantes a ficarem **estritamente a zero**.
  * Realiza **seleção automática de variáveis**, produzindo modelos esparsos e interpretáveis.
* **Ridge Regression (Regularização L2):**
  $$\text{Penalização} = \lambda \sum_{j=1}^{p} \beta_j^2$$
  * Encolhe (*shrinks*) a magnitude dos coeficientes aproximando-os de zero, mas **nunca os anula por completo**.
  * Mantém todas as variáveis, sendo eficaz quando há forte colinearidade entre atributos.

---

## 6. Importância Crítica do Escalonamento (*Scaling*)

Em algoritmos otimizados por Gradiente Descendente, variáveis com ordens de grandeza muito díspares (ex.: Idade $[18-80]$ vs. Rendimento Anual $[10.000-200.000]$) criam curvas de nível elípticas e distorcidas na função de custo.

* **Impacto sem escalonamento:** O gradiente oscila transversalmente, atrasando drasticamente a convergência ou gerando divergência numérica.
* **Solução:** Aplicar sempre **Padronização ($Z$-Score: $\mu=0, \sigma=1$)** ou **Normalização (Min-Max: $[0, 1]$)** nas variáveis independentes antes de iniciar a otimização por gradiente.




# Prática

## PARTE 1: Regressão Linear na Prática (Previsão Contínua)

**Problema:** Prever o valor de venda de imóveis (`Price`) com base em características socioeconómicas e demográficas da região (*USA Housing*).

```
   Dataset Bruto                  EDA & Limpeza                Modelagem                Diagnóstico
[USA_Housing.csv]  ──>  [Remover 'Address']  ──>  [LinearRegression]  ──>  [Resíduos ~ Normal?]
(5000 instâncias)       [Pairplot & Heatmap]      [Treino/Teste 60/40]     [MAE, MSE, RMSE]
```

### 1. Análise Exploratória (EDA) & Seleção de Variáveis
* **Inpeção Estrutural:** `df.info()` e `df.describe()` revelam 5.000 registos sem valores nulos em 7 colunas.
* **Inspeção Visual:**
  * `sns.pairplot(df, hue='Price')`: Revela correlações lineares fortes (especialmente entre `Avg. Area Income` e `Price`).
  * `sns.heatmap(df.corr(numeric_only=True))`: Confirma as variáveis independentes com maior correlação linear com o alvo.
  * `sns.histplot(df['Price'])`: Confirma que a variável alvo segue uma distribuição aproximadamente Normal/Gaussiana.
* **Tratamento de Atributos:** O atributo `Address` é removido por se tratar de texto sem estrutura numérica ou categórica direta útil a nível básico.

```python
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```

---

### 2. Treino e Interpretação dos Coeficientes

```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
```

A interpretação dos coeficientes deve ser feita sob o princípio **_ceteris paribus_** (mantendo todas as restantes variáveis constantes):

| Variável | Coeficiente ($\beta_j$) | Interpretação Prática |
| :--- | :---: | :--- |
| `Avg. Area Income` | $+21.53$ | Cada aumento de $1 na renda média da área aumenta o preço em **$21.53**. |
| `Avg. Area House Age`| $+164,883.28$ | Cada ano adicional de idade média das casas na zona valoriza a casa em **$164,883.28**. |
| `Avg. Area Number of Rooms` | $+122,368.68$ | Cada divisão extra em média na zona valoriza a casa em **$122,368.68**. |
| `Avg. Area Number of Bedrooms` | $+2,233.80$ | Cada quarto extra em média tem um impacto marginal modesto de **$2,233.80**. |
| `Area Population` | $+15.15$ | Cada habitante adicional na cidade aumenta o preço em **$15.15**. |
| *Intercepto ($\beta_0$)* | $-2,640,159.80$ | Valor base quando todos os atributos são nulos. |

---

### 3. Diagnóstico do Modelo & Validação de Resíduos

Antes de aceitar as métricas de erro, valida-se o comportamento dos resíduos ($e_i = y_i - \hat{y}_i$):
1. **Gráfico de Dispersão ($y_{\text{real}}$ vs. $\hat{y}_{\text{pred}}$):** Os pontos devem distribuir-se ao longo de uma reta diagonal uniforme. Desvios curvos denunciam relações não-lineares ignoradas.
2. **Histograma dos Resíduos:** A diferença $(y_{\text{test}} - \hat{y})$ tem de apresentar uma **distribuição Normal centrada em zero**. Assimetrias acentuadas indicam viés de previsão ou presença de *outliers* graves.

```python
from sklearn import metrics
import numpy as np

mae = metrics.mean_absolute_error(y_test, predictions)   # 82,288.22$
mse = metrics.mean_squared_error(y_test, predictions)    # 10,460,958,907.21$^2
rmse = np.sqrt(mse)                                      # 102,278.83$
```

---

## PARTE 2: Regularização — Regressão Linear vs. LASSO ($L_1$)

A regularização introduz uma penalização para conter coeficientes sobredimensionados e mitigar o sobreajuste (*overfitting*).

### Experiência Prática: Injeção de Ruído
Adicionam-se colunas puramente aleatórias (`randomCol1`, `randomCol2`) ao dataset.

```python
from sklearn.linear_model import Lasso

# Modelo sem regularização
lin_reg = LinearRegression().fit(X_train, y_train)

# Modelo com regularização L1
lasso = Lasso(alpha=2000, random_state=2025).fit(X_train, y_train)
```

### Comparação dos Coeficientes:

$$\begin{array}{l|c|c|l}
\textbf{Atributo} & \textbf{Regressão Linear} & \textbf{LASSO} & \textbf{Efeito Prático} \\ \hline
\text{Avg. Area Income} & 21.58 & 21.58 & \text{Mantido} \\
\text{Avg. Area House Age} & 165,790.06 & 163,742.88 & \text{Suave encolhimento} \\
\text{randomCol1 (Ruído)} & \mathbf{220.94} & \mathbf{0.00} & \textbf{Eliminado pelo LASSO} \\
\text{randomCol2 (Ruído)} & 134.50 & 125.11 & \text{Encolhido, mas retido}
\end{array}$$

* **Conclusão:** O LASSO anulou com sucesso a variável ruidosa `randomCol1`, demonstrando a sua capacidade nativa de **seleção automática de atributos**. 
* *Nota Técnica:* Se os dados tiverem escalas muito díspares ou o parâmetro $\alpha$ for desadequado, o LASSO pode não anular todo o ruído (como aconteceu com `randomCol2`), tornando imperativo o escalonamento (*standardization*).

---

## PARTE 3: Regressão Logística na Prática (Classificação Binária)

**Problema:** Prever a sobrevivência de passageiros do naufrágio do Titanic (`Survived`: $0$ ou $1$).

---

### 1. Limpeza Estratégica e Imputação Condicional

A análise de valores em falta (`sns.heatmap(train.isnull())`) revelou dois cenários distintos:
* **Coluna `Cabin`:** Mais de 70% de dados em falta $\to$ **Remover** ou binarizar (*"Cabine Conhecida: 1 ou 0"*).
* **Coluna `Age`:** ~20% de nulos $\to$ **Imputar**.

#### Imputação Condicional (Mediana por Classe)
A idade correlaciona-se com a riqueza do passageiro: passageiros de 1ª classe eram tendencialmente mais velhos do que os da 3ª classe (`sns.boxplot(x='Pclass', y='Age')`).  
**Solução avançada:** Imputar a mediana da classe em vez da média global:

```python
train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('median'))
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True) # Descarta apenas as 2 linhas nulas de 'Embarked'
```

---

### 2. Codificação Categórica (*Dummy Variables*)

Modelos lineares exigem entradas numéricas. As variáveis de texto nominais (`Sex`, `Embarked`) são convertidas em colunas binárias. Para evitar a armadilha da multicolinearidade (*Dummy Variable Trap*), remove-se a primeira categoria de referência:

```python
sex = pd.get_dummies(train['Sex'], drop_first=True)         # Gera coluna 'male' (0=Feminino, 1=Masculino)
embark = pd.get_dummies(train['Embarked'], drop_first=True) # Gera colunas 'Q' e 'S' (referência: 'C')

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)
```

---

### 3. Guia Prático de Otimizadores (*Solvers*) no Scikit-Learn

A classe `LogisticRegression` suporta múltiplos algoritmos de otimização numérica para convergir à solução de máxima verosimilhança:

| *Solver* | Penalizações Suportadas | Escala de Dados Recomendada | Multiclasse Nativo (*Multinomial*) |
| :--- | :--- | :--- | :---: |
| **`liblinear`** | **$L_1$, $L_2$** | **Datasets Pequenos** | Não (Apenas *One-vs-Rest*) |
| **`lbfgs`** *(default)* | $L_2$, None | Médios / Grandes | Sim |
| **`newton-cg`** | $L_2$, None | Médios / Grandes | Sim |
| **`sag`** | $L_2$, None | **Datasets Muito Grandes** | Sim |
| **`saga`** | **ElasticNet, $L_1$, $L_2$, None** | **Datasets Muito Grandes** | Sim |

#### Benchmark Empírico (Caso Titanic):
* **`liblinear`:** Extremamente veloz em amostras pequenas (~800 linhas).
* **`newton-cg`:** Rápido, mas com custo por iteração mais elevado.
* **`lbfgs`:** Excelente equilíbrio e precisão de convergência.

---

### 4. Avaliação do Classificador

A avaliação compara o desempenho entre os *solvers* e classes através de relatórios de métricas e matrizes de confusão:

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Relatório detalhado
print(classification_report(y_test, predictions))

# Matriz de Confusão Visual
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.show()
```

#### Interpretação do Relatório de Classificação:
* **Classe 0 (Não Sobreviveu):** Revocação (*Recall*) elevada (~$0.91$), indicando que o modelo identifica corretamente a maioria das mortes.
* **Classe 1 (Sobreviveu):** Precisão elevada (~$0.84$), significando que quando o modelo prevê que alguém sobreviveu, tem alta taxa de acerto; contudo, o *Recall* ronda os ~$0.68$, indicando que falha em identificar cerca de 32% dos sobreviventes reais (Falsos Negativos).
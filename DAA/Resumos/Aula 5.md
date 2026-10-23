# Teórica
## 1. Fundamentos & Taxonomia dos *Ensembles*

O objetivo do *Ensemble Learning* é combinar múltiplos modelos base (*base learners* ou *weak learners*) para construir um preditor global mais robusto, reduzindo o **viés (*bias*)**, a **variância (*variance*)** ou ambos.

```
                                  ENSEMBLE LEARNING
         _________________________________|_________________________________
        |                                 |                                 |
1. Paralelos (Bagging)          2. Sequenciais (Boosting)        3. Meta-Modelos (Stacking)
- Explora a INDEPENDÊNCIA       - Explora a DEPENDÊNCIA          - Explora a HETEROGENEIDADE
- Reduz a VARIÂNCIA             - Reduz o VIÉS                   - Combina diferentes famílias
- Ex.: Random Forest            - Ex.: AdaBoost, XGBoost         - Ex.: Stacking, Blending
```

> **Regra de Ouro da Diversidade:**  
> Combinar modelos semelhantes e com elevado desempenho traz ganhos marginais quase nulos. A força do *ensemble* reside em combinar modelos **diversificados e heterogéneos**, de modo a que os erros cometidos por um modelo sejam compensados pelos acertos dos outros.

---

## 2. Métodos de Votação e Agregação Simples

Técnicas que combinam predições de vários modelos previamente treinados de forma independente.

### A. Votação Máxima (*Max Voting*) — Classificação
* **Votação Rígida (*Hard Voting*):** Aplica a regra da maioria sobre as classes previstas.
  * *Exemplo:* Modelo 1 prevê $\text{A}$; Modelo 2 prevê $\text{B}$; Modelo 3 prevê $\text{B} \implies$ **Decisão final: $\text{B}$** ($2/3$ dos votos).
* **Votação Suave (*Soft Voting*):** Calcula a média das probabilidades de pertença a cada classe.
  * *Exemplo:*
    * Modelo 1 prevê $\text{Classe A}$ com probabilidade de $99\%$.
    * Modelo 2 prevê $\text{Classe A}$ com probabilidade de $49\%$ (favorece B por $51\%$).
    * Modelo 3 prevê $\text{Classe A}$ com probabilidade de $49\%$ (favorece B por $51\%$).
    * *Média da Classe A:* $\frac{99\% + 49\% + 49\%}{3} = \mathbf{65.67\%}$.
    * **Decisão final: $\text{Classe A}$**.  
    *(O Soft Voting é frequentemente superior ao Hard Voting porque pondera o nível de certeza/confiança de cada modelo).*

```python
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(
    estimators=[('svm', best_svm), ('xgb', best_xgb), ('rfc', best_rfc)],
    voting='soft',
    weights=[1, 1, 1]  # Permite atribuir pesos superiores aos melhores modelos
)
vc.fit(X_train, y_train)
```

---

### B. Média Simples e Ponderada (*Averaging*) — Regressão
* **Média Simples:** $\hat{y}_{\text{final}} = \frac{1}{M}\sum_{m=1}^{M} \hat{y}_m$
* **Média Ponderada:** Dá maior relevância aos modelos com menor erro no conjunto de validação:
  $$\hat{y}_{\text{final}} = \sum_{m=1}^{M} w_m \hat{y}_m \quad \left(\text{onde } \sum w_m = 1\right)$$

---

## 3. Família Paralela: *Bagging* & *Random Forest*

O *Bagging* (**B**ootstrap **Agg**regat**ing**) treina múltiplos modelos de forma paralela e independente para **diminuir a variância** (combater o sobreajuste de modelos instáveis, como árvores de decisão profundas).

```
Dataset Original (D) ──> Amostragem com Reposição (Bootstrapping) ──> d1, d2, d3, ...
                                                                          │   │   │
Treino Independente em Paralelo ──────────────────────────────────────>  M1  M2  M3
                                                                          │   │   │
Agregação Final (Votação na Classificação / Média na Regressão) ──────> [ Predição Final ]
```

### A. Random Forest: O Duplo Nível de Aleatoriedade
O Random Forest melhora o Bagging tradicional ao introduzir duas fontes de aleatoriedade que descorrelacionam as árvores:
1. **Amostragem de Instâncias (*Bootstrapping*):** Cada árvore é treinada numa amostra aleatória com reposição do dataset original.
2. **Amostragem de Atributos (*Feature Sub-spacing*):** Em cada nó e divisão da árvore, apenas um subconjunto aleatório de atributos é avaliado (tipicamente $\mathbf{\sqrt{n}}$ atributos em classificação, onde $n$ é o total de variáveis).

### B. Avaliação Nativa: *Out-Of-Bag* (OOB) Error
* No processo de amostragem com reposição, matematicamente cerca de **$1/3$ das amostras ($\approx 36.8\%$)** nunca é selecionado para uma dada árvore — este subconjunto constitui os dados **Out-Of-Bag**.
* **OOB Error:** Cada instância do dataset é avaliada usando apenas as árvores que *não* a utilizaram no treino. Permite estimar o erro de generalização **sem necessidade de validação cruzada externa adicional**.

---

## 4. Família Sequencial: *Boosting*

O *Boosting* treina modelos de forma sequencial, onde cada novo modelo é desenhado para **corrigir os erros cometidos pelos modelos anteriores**, reduzindo primariamente o **viés (*bias*)**.

```
[ Modelo 1 ] ──> Identifica Erros ──> [ Pondera Pontos Difíceis ] ──> [ Modelo 2 ] ──> ...
```

---

### A. A Linha Evolutiva do *Boosting*

```
Árvores de Decisão ──> Bagging ──> Random Forest ──> Boosting (AdaBoost) ──> Gradient Boosting ──> XGBoost
 (Base simples)     (Paralelo)   (Sub-atributos)   (Pesos nas instâncias)   (Descida Gradiente)   (Otimizado/Regularizado)
```

| Algoritmo | Mecanismo de Funcionamento | Principais Características |
| :--- | :--- | :--- |
| **AdaBoost** <br>(*Adaptive Boosting*) | Atribui pesos superiores às instâncias classificadas incorretamente. As árvores seguintes focam-se nos dados errados. | Combina modelos fracos (*weak learners / stumps*); poucos hiperparâmetros; sensível a *outliers*. |
| **Gradient Boosting** | Em vez de ajustar pesos nas instâncias, treina o novo modelo sobre os **resíduos (erros)** do anterior usando **Gradiente Descendente**. | Otimiza qualquer função de perda diferenciável; alto poder preditivo; treino sequencial mais lento. |
| **XGBoost** <br>(*Extreme Gradient Boosting*) | Implementação de alto desempenho do Gradient Boosting com engenharia de hardware avançada. | **Regularização ($L_1$ e $L_2$)** nos nós; processamento em paralelo; poda automática (*tree pruning*); gestão nativa de valores em falta. |

```python
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)

# XGBoost com Regularização e Restrições de Complexidade
model_xgb = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    colsample_bytree=0.8,  # Sub-amostragem de atributos estilo Random Forest
    objective='binary:logistic'
)
```

---

## 5. Meta-Modelos: *Stacking* vs. *Blending*

Em vez de agregar por votação simples, estes métodos treinam um **segundo modelo (Meta-Learner)** que aprende a ponderar e combinar as previsões de múltiplos modelos base heterogéneos.

---

### A. *Stacking* (*Stacked Generalization*)

```
            DADOS DE ENTRADA (X)
             /       |        \
      [Modelo 1] [Modelo 2] [Modelo 3]      <-- Nível 0 (Modelos Heterogéneos: RF, SVM, XGB)
             \       |        /
       [ Pred_1   Pred_2   Pred_3 ]         <-- Meta-Atributos (Meta-Features)
                     │
              [META-LEARNER]                <-- Nível 1 (ex.: Regressão Logística)
                     │
             [PREVISÃO FINAL]
```

1. **Nível 0 (*Base Learners*):** Múltiplos algoritmos de famílias diferentes (ex.: Random Forest, SVM, k-NN, Redes Neuronais) geram previsões através de validação cruzada (*out-of-fold*).
2. **Nível 1 (*Meta-Learner*):** As previsões geradas pelo Nível 0 servem como novas variáveis de entrada (*meta-features*) para treinar o modelo final (geralmente uma Regressão Logística ou Linear simples).

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svm', make_pipeline(StandardScaler(), LinearSVC()))
]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf.fit(X_train, y_train)
```

---

### B. *Blending* (A Variação por Conjunto de Validação)

O *Blending* segue a mesma lógica do *Stacking*, mas simplifica a geração das *meta-features*:
1. Os dados de treino são divididos numa partição fixa de **Treino** e outra de **Validação (*Hold-Out*)**.
2. Os modelos base de Nível 0 são treinados **apenas** no conjunto de Treino.
3. As previsões desses modelos sobre o conjunto de **Validação** tornam-se o dataset de treino do **Meta-Modelo**.
* *Compromisso:* Mais simples e computacionalmente mais rápido do que o *Stacking* com validação cruzada, mas descarta parte dos dados de treino dos modelos base e do meta-aprendiz.

---

## 6. Quadro Comparativo Global

| Estratégia | Treino dos Modelos | Foco Principal | Redução do Erro | Modelos Base Recomendados |
| :--- | :--- | :--- | :---: | :--- |
| **Max Voting / Averaging** | Paralelo / Independente | Simplicidade e consenso | Variância | Modelos robustos e calibrados |
| **Bagging / Random Forest** | **Paralelo** (com *Bootstrap*) | Combater o sobreajuste | **Variância** | Modelos instáveis e complexos (Árvores profundas) |
| **Boosting (AdaBoost/XGB)** | **Sequencial** (correção de erros) | Aumentar a precisão | **Viés** e Variância | Modelos simples e fracos (*Stumps* ou árvores rasas) |
| **Stacking / Blending** | Hierárquico (Nível 0 $\to$ Nível 1) | Otimizar a combinação | Ambos | **Modelos Heterogéneos** (SVM + Árvores + Redes) |

# Prática

## 1. Engenharia e Preparação de Atributos (*Feature Engineering*)

A qualidade das variáveis de entrada determina o teto de desempenho de qualquer algoritmo preditivo.

```
Dados Brutos ──> [Filtro de Nulos] ──> [Filtro de Variância Zero] ──> [Codificação Categórica] ──> [Decomposição Temporal]
```

---

### A. Gestão de Valores em Falta (*Missing Values*)
A estratégia depende de estarmos perante atributos preditores ($X$) ou da variável alvo ($y$):
* **No Alvo (`incidents` com ~40% nulos):** **Eliminação obrigatória das linhas.** Em aprendizagem supervisionada, observações sem rótulo não podem ser usadas no treino nem na validação (`incid.dropna(inplace=True)`).
* **Nos Preditores (`affected_roads` com 85 nulos):** Quando a variável apresenta elevada complexidade textual e reduzido valor preditivo adicional, a remoção da coluna (`drop`) evita ruído desnecessário.

---

### B. Remoção de Atributos com Variância Zero (*Unique Values*)
Variáveis que assumem um **único valor constante** em todas as linhas têm variância nula ($\sigma^2 = 0$) e poder discriminativo zero:
* `city_name`: Todas as entradas são `"Guimarães"`. $\implies$ **Remover coluna.**
* `avg_precipitation`: Todas as entradas são `0.0`. $\implies$ **Remover coluna.**

---

### C. Engenharia de Atributos Temporais (*Handling Dates*)
Modelos matemáticos não interpretam objetos textuais de data diretamente. O atributo `record_date` (`"2021-03-15 23:00"`) deve ser convertido para `datetime` e decomposto:

```python
incid['record_date'] = pd.to_datetime(incid['record_date'], format='%Y-%m-%d %H:%M')

# Extração de componentes cíclicos e sazonais
incid['month'] = incid['record_date'].dt.month
incid['day']   = incid['record_date'].dt.day
incid['hour']  = incid['record_date'].dt.hour
```

* **Regra de Poda Imediata:** Se após a extração se verificar que `year` (todos em 2021) e `minute` (todos a 0) são constantes, **devem ser eliminados de imediato**.
* **Variáveis Derivadas Avançadas:** Cálculo de dias decorridos face a uma data de referência (`(today - record_date).dt.days`) ou sinalizadores de dia útil vs. fim de semana.

---

## 2. Guia de Codificação Categórica (*Encoding*)

Transformação de variáveis nominais/ordinais em matrizes numéricas utilizáveis:

| Técnica | Funcionamento | N.º de Colunas Geradas | Quando Utilizar |
| :--- | :--- | :---: | :--- |
| **Factorizing / Label Encoding** | Atribui inteiros sequenciais ($1, 2, 3, \dots$) via `pd.factorize()`. | $1$ | Árvores de decisão e variáveis **ordinais** com hierarquia natural (*Low, Medium, High*). |
| **One-Hot Encoding** | Cria uma coluna binária ($0$ ou $1$) para cada categoria distinta. | $K$ (número de classes) | Variáveis nominais com **baixa cardinalidade** (poucas categorias). |
| **Binary Encoding** | Converte as categorias em código binário decimal reduzido (exige `category_encoders`). | $\approx \lceil \log_2(K) \rceil$ | Variáveis nominais com **alta cardinalidade** (evita explosão de dimensões). |
| **Backward Difference** | Codificação por contraste que normaliza as transições entre níveis de $-1$ a $+1$. | $K - 1$ | Análise de regressão onde importa o efeito diferencial entre categorias ordenadas. |

---

## 3. Parâmetros vs. Hiperparâmetros

A distinção entre o que o modelo aprende e o que o engenheiro configura:

```
                            CONFIGURAÇÕES DE UM MODELO
                       __________________|__________________
                      |                                     |
             PARÂMETROS DO MODELO                 HIPERPARÂMETROS
        - Estimados durante o treino           - Definidos externamente ANTES do treino
        - Não são ajustados à mão              - Afinados por experimentação / GridSearch
        - Ex.: Pesos numa Rede Neuronal        - Ex.: Learning rate (α), Regularização (C, λ)
        - Ex.: Limiares de corte numa Árvore   - Ex.: max_depth, criterion, ccp_alpha
```

---

## 4. Otimização com Validação Cruzada (*GridSearchCV*)

Treinar uma árvore sem restrições (`max_depth=None`) faz com que ela cresça até à pureza absoluta. No caso dos dados de Guimarães, gerou uma árvore descontrolada com **profundidade 24 e 1.189 folhas**, memorizando o ruído e limitando a exatidão no teste a $62\%$.

O `GridSearchCV` atua como um **meta-estimador**: executa uma busca exaustiva sobre uma grelha de combinações candidatas avaliadas por Validação Cruzada ($k\text{-Fold}$):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=2022),
    param_grid=param_grid,
    cv=5,
    refit=True  # Retreina automaticamente o melhor modelo sobre todo o conjunto de treino
)
grid_dt.fit(X_train, y_train)

print(grid_dt.best_params_)  # {'criterion': 'entropy', 'max_depth': 6}
```

* **Resultado:** Reduzir a profundidade máxima de 24 para **6** aumentou a exatidão no conjunto de teste de $62\%$ para **$64\%$**, criando um modelo incomparavelmente mais simples e interpretável.

---

## 5. Poda de Árvores (*Decision Tree Pruning*)

A poda substitui sub-árvores complexas por nós folha sempre que o erro esperado da sub-árvore não compense a sua complexidade.

```
       [ Árvore Totalmente Desenvolvida ]
              (Profundidade 24, 1189 folhas)
                     │
         PÓS-PODA POR CUSTO-COMPLEXIDADE (ccp_alpha)
                     │
       [ Árvore Ótima Podada ]
              (Generalização máxima no Teste)
```

### A. Pré-Poda (*Early Stopping*)
Trava o crescimento antes da convergência total:
* `max_depth`: Limita o teto de ramificações verticais.
* `min_samples_split`: Mínimo de instâncias exigidas para autorizar nova divisão.
* `min_samples_leaf`: Mínimo de instâncias que uma folha terminal deve conter.

---

### B. Pós-Poda por Custo-Complexidade (*Cost-Complexity Pruning / ccp_alpha*)
Permite o crescimento total e remove ramos de baixo para cima minimizando a função de custo-complexidade $R_\alpha(T)$:

$$R_\alpha(T) = R(T) + \alpha |T|$$

* $R(T)$: Erro de treino (impureza total ou SSR).
* $|T|$: Número de folhas terminais (penalização pela complexidade).
* $\alpha$ (`ccp_alpha`): Fator de penalização.
  * Se $\alpha = 0$: Nenhuma poda (árvore original completa).
  * À medida que $\alpha$ cresce: Árvores mais pequenas e generalistas são favorecidas.

#### Implementação Prática com Scikit-Learn:

```python
# 1. Extração da trajetória de alphas eficazes
path = dt_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 2. Seleção do alpha ótimo via GridSearchCV
param_grid_pru = {'ccp_alpha': ccp_alphas}
ccp_alpha_gs = GridSearchCV(DecisionTreeClassifier(random_state=2022), param_grid_pru, cv=5)
ccp_alpha_gs.fit(X_train, y_train)

best_tree = ccp_alpha_gs.best_estimator_
```

---

### C. A Curva de Compromisso Viés-Variância no $\alpha$

```
Exatidão ^
         │        Curva de Treino (Cai monotonicamente à medida que se poda)
    1.00 ├──────\
         │       \
         │        \          Ponto Ótimo (ccp_alpha)
         │         \            o  <-- Máximo de Exatidão no Teste
    0.93 ├──────────\──────────/ \
         │           \________/   \  Curva de Teste (Generalização)
         │                         \
    0.88 ├──────────────────────────\──────>
         0.000      0.008      0.012      0.035   ccp_alpha
       (Sem poda)                         (Poda excessiva / Underfitting)
```

1. **$\alpha$ muito baixo:** A árvore de teste tem baixo desempenho devido a sobreajuste (*overfitting*).
2. **$\alpha$ ótimo:** Ponto em que a árvore descarta o ruído da amostra e atinge o pico de generalização nos dados de teste.
3. **$\alpha$ excessivo:** A árvore perde nós críticos, caindo em subajuste (*underfitting*).


# Teórica
## MÓDULO 1: Fundamentos e Paradigmas de Machine Learning

### 1. Machine Learning vs. Ciência de Dados

Embora partilhem ferramentas, têm propósitos e entregáveis distintos:

| Critério           | Aprendizagem Automática (*Machine Learning*)                                  | Ciência de Dados (*Data Science*)                                                      |
| :----------------- | :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| **Foco principal** | Algoritmos e modelos que aprendem padrões autonomamente a partir de dados.    | Estudo exploratório e metodológico para extrair conhecimento e significado do negócio. |
| **Produto Final**  | **Artefacto de software** (modelo em produção, API de inferência).            | **Relatórios, *dashboards* e apresentações executivas**.                               |
| **Papel típico**   | *ML Engineer* / *Data Engineer* (desenvolve e implementa modelos escaláveis). | *Data Scientist* (formula hipóteses, desenha metodologias e analisa teorias).          |

### 2. Mudança de Paradigma: Programação Clássica vs. Machine Learning

* **Programação Tradicional:** $\text{Dados} + \text{Regras/Programa} \longrightarrow \mathbf{Resultados}$
* **Machine Learning (Fase de Treino):** $\text{Dados} + \text{Resultados Esperados} \longrightarrow \mathbf{Modelo}$
* **Machine Learning (Fase de Inferência):** $\text{Novos Dados} + \mathbf{Modelo} \longrightarrow \mathbf{Previsões}$

### 3. Aprendizagem Simbólica vs. Não Simbólica

* **Simbólica (Explicável / *White-Box*):**
  * *Base:* Lógica formal, regras explícitas de leitura humana e grafos de procura (ex.: Árvores de Decisão, Raciocínio Baseado em Casos).
  * *Vantagem:* Fácil auditoria e interpretação do raciocínio.
  * *Desvantagem:* Exige formulação manual ou computacionalmente pesada de regras; pouca escalabilidade em tarefas percetuais complexas.
* **Não Simbólica (Conexionista / *Black-Box*):**
  * *Base:* Cálculos numéricos, matrizes de pesos e otimização contínua (ex.: Redes Neuronais Artificiais, Deep Learning, SVM).
  * *Vantagem:* Extrai padrões abstratos de dados não estruturados (áudio, imagem, texto).
  * *Desvantagem:* Difícil explicar matematicamente *o porquê* de uma decisão — sensível em áreas críticas (medicina, condução autónoma).

### 4. Os Três Grandes Paradigmas de Aprendizagem

```
                            APRENDIZAGEM AUTOMÁTICA
       ________________________________|________________________________
      |                                |                                |
Supervisionada                 Não Supervisionada                   Por Reforço
(Dados rotulados: X -> y)      (Apenas dados X)                (Agente + Recompensa/Penalização)
  |-- Classificação (Discreto)   |-- Clustering (Agrupamento)    |-- Q-Learning
  `-- Regressão (Contínuo)       |-- Redução de Dimensionalidade `-- SARSA
                                 `-- Associação
```

#### A. Aprendizagem Supervisionada ($y = f(x)$)
Exige pares $(x_i, y_i)$. O modelo afina os parâmetros minimizando o erro entre o valor real e a previsão:
* **Classificação (Alvo Discreto / Classes):** Deteção de fraude, diagnóstico clínico, classificação de imagens, *churn* (fidelização de clientes).
* **Regressão (Alvo Numérico Contínuo):** Previsão de preços no mercado, estimativa da esperança média de vida, meteorologia, crescimento populacional.

#### B. Aprendizagem Não Supervisionada
Não existem rótulos $y$. Modela a estrutura intrínseca dos dados:
* **Segmentação (*Clustering*):** Agrupar clientes por perfil de compra, sistemas de recomendação.
* **Redução de Dimensionalidade:** Condensação de variáveis para visualização e compressão (ex.: PCA).
* **Associação:** Regras de coocorrência (ex.: análise do cesto de compras — *"quem compra X também compra Y"*).

#### C. Aprendizagem por Reforço (*Reinforcement Learning*)
Não tem supervisor com respostas certas nem opera a frio sobre dados estáticos. Aprende por **tentativa e erro** mediante um sinal avaliativo (*feedback* de recompensas e penalizações):
* **Algoritmos clássicos:** Q-Learning (baseado em política ótima direta) e SARSA (baseado na política seguida).
* **Aplicações:** Robótica autónoma, agentes de jogos (xadrez, Go, videojogos), decisões operacionais em tempo real.

## MÓDULO 2: Metodologias e Pipelines de Projeto

Para garantir replicabilidade, controlo de custos e resultados fiáveis, a indústria adota normas estruturadas:

### 1. Metodologias Padronizadas

* **CRISP-DM (*Cross Industry Standard Process for Data Mining*):** A metodologia mais adotada globalmente. Ciclo iterativo com 6 fases:
  1. *Business Understanding* (Definição dos objetivos de negócio)
  2. *Data Understanding* (Exploração inicial e qualidade dos dados)
  3. *Data Preparation* (Limpeza, transformação, enriquecimento)
  4. *Modeling* (Treino de modelos e parametrização)
  5. *Evaluation* (Avaliação técnica e de alinhamento com o negócio)
  6. *Deployment* (Colocação em produção e integração)
* **SEMMA (SAS):** $\text{Sample} \to \text{Explore} \to \text{Modify} \to \text{Model} \to \text{Assess}$.

### 2. Fluxo Prático da Pipeline de ML (*End-to-End*)

$$\text{Problema} \to \text{Ingestão} \to \text{Preparação} \to \text{Segregação (Treino/Val/Teste)} \to \text{Treino} \to \text{Avaliação} \to \text{Deploy} \to \text{Monitorização}$$

* **Segregação Obrigatória:** Divisão estrita em **Conjunto de Treino** (ajuste de pesos), **Validação** (afinação de hiperparâmetros) e **Teste** (avaliação final cega contra sobreajuste/*overfitting*).

## MÓDULO 3: Qualidade e Exploração dos Dados (EDA)

> *«Limpar e preparar dados consome cerca de 60% a 80% do tempo de um projeto de Data Science.»*

### 1. Patologias Típicas dos Dados & O Viés de Sobrevivência
* **Viés de Sobrevivência (Lição de Abraham Wald na WWII):** Analisar apenas as aeronaves que regressavam cheias de marcas de balas sugeria reforçar essas áreas. Wald provou o inverso: era imperativo reforçar as áreas **sem** marcas de balas, pois os aviões atingidos nesses pontos fulcrais tinham-se despenhado e não constavam da amostra. Em ML: *cuidado com dados truncados ou amostras enviesadas*.
* **Ruído (*Noise*):** Flutuações indesejadas por erro de medição, erros humanos ou avarias de sensores.
* **Valores em Falta (*Missing Values*):** Omissões por falta de resposta, dados sensíveis ocultados ou variáveis não aplicáveis.
* **Duplicados:** Registos repetidos que sobreponderam observações de forma espúria.

### 2. Ferramentas de Análise Exploratória (EDA)
* **Tabelas de Contingência & Teste Qui-Quadrado ($\chi^2$):** Avaliam a independência entre duas variáveis categóricas (ex.: avaliar se o género influencia o sentimento de compra).
* **Matriz de Correlação:** Mede relações lineares entre atributos numéricos. Identifica se existem variáveis preditoras redundantes ou altamente associadas ao *target*.
* **Visualização:** Histogramas (distribuições), *Scatter Plots* (dispersão bivariada) e *Boxplots* (deteção visual de assimetrias e valores extremos).

## MÓDULO 4: Preparação e Engenharia de Atributos (*Data Prep*)

### 1. Escalonamento de Variáveis (*Feature Scaling*)

* **Porquê?** Algoritmos baseados em distâncias (k-NN, SVM, K-Means) ou gradientes são fortemente dominados por variáveis com ordens de grandeza elevadas (ex.: Salário em € vs. Idade em anos).

| Método | Fórmula | Quando Usar |
| :--- | :---: | :--- |
| **Normalização (Min-Max)** | $z = \frac{x - \min(x)}{\max(x) - \min(x)}$ | Comprime os dados para o intervalo $[0, 1]$. Recomendado quando os dados **não seguem** uma distribuição Gaussiana. |
| **Padronização (Z-Score)** | $z = \frac{x - \mu}{\sigma}$ | Converte para média $\mu=0$ e desvio padrão $\sigma=1$. Ideal para dados que seguem uma distribuição aproximadamente Gaussiana. |

### 2. Gestão de *Outliers*

* **Deteção:**
  1. *Estatística:* Z-score ($|z| > 3$) ou Critério do Boxplot ($x < Q_1 - 1.5\text{IQR}$ ou $x > Q_3 + 1.5\text{IQR}$).
  2. *Conhecimento de Domínio:* Regras de negócio rígidas (ex.: salário mensal $> 1.000.000\,€$).
  3. *Baseada em Modelos:* *Isolation Forest*, *One-Class SVM*.
* **O Dilema — Eliminar (*Drop*) vs. Limitar (*Cap*):**
  * **Drop:** Elimina o registo. Garante pureza, mas reduz a dimensão da amostra.
  * **Cap / Winsorization:** Trunca o valor extremo no limite superior/inferior. Preserva o tamanho do dataset, mas deforma as caudas da distribuição original.

### 3. Seleção de Variáveis (*Feature Selection*)

Reduzir variáveis irrelevantes ou redundantes combate a **maldição da dimensionalidade**, mitiga o **overfitting**, acelera o treino e clarifica a interpretabilidade.

* **Heurísticas Rápidas de Eliminação:**
  * Percentagem de valores nulos acima de um limiar crítico (ex.: $> 60\%$).
  * Variância muito baixa / desvio padrão próximo de zero (atributo quase constante).
  * Variáveis com multicolinearidade extrema (correlação quase perfeita entre duas variáveis explicativas).
* **As Três Famílias de Métodos:**
  1. **Filtro (*Filter Methods*):** Métricas estatísticas aplicadas antes do modelo (ex.: correlação, teste $\chi^2$). Rápidos e independentes do algoritmo.
  2. **Invólucro (*Wrapper Methods*):** Usa o próprio algoritmo para testar subconjuntos de variáveis (ex.: *Sequential Forward Selection* ou *Backward Elimination*). Mais precisos, mas computacionalmente exigentes.
  3. **Embutidos (*Embedded Methods*):** A seleção é intrínseca ao treino do algoritmo (ex.: Regularização L1 / Lasso, que zera os pesos de variáveis inúteis; árvores de decisão).
* **Redução Dimensional Não Supervisionada:** **PCA (*Principal Component Analysis*)**, que projeta o espaço original num novo conjunto ortogonal de componentes que retêm a máxima variância dos dados.

### 4. Tratamento de Valores em Falta (*Missing Values*)

1. **Remoção:** Eliminação da coluna (se tiver muitos nulos) ou de linhas (se as instâncias incompletas forem residuais).
2. **Imputação Estatística:** Substituição por Média/Mediana (dados numéricos) ou Moda (dados categóricos).
3. **Interpolação / Modelos:** Estimação baseada na tendência ou via modelos como k-NN Imputer.
4. **Máscara (*Missing Indicator*):** Imputação de um valor fixo combinada com uma coluna binária adicional que assinala se o dado original estava em falta.

### 5. Codificação de Variáveis Categóricas (*Encoding*)

| Técnica | Funcionamento | Prós / Contras |
| :--- | :--- | :--- |
| **Label Encoding** | Atribui um inteiro consecutivo a cada classe ($0, 1, 2\dots$). | **Pró:** Compacto. <br>**Contra:** Só deve ser usado em dados **ordinais**. Se usado em nominais, os modelos interpretam uma relação métrica/ordem inexistente ($2 > 1 > 0$). |
| **One-Hot Encoding** | Cria uma coluna binária ($0$ ou $1$) por cada categoria distinta. | **Pró:** Não assume qualquer relação de ordem (ideal para nominais). <br>**Contra:** Sofre com alta cardinalidade (cria demasiadas colunas esparsas). |
| **Binary Encoding** | Codifica as categorias em formato binário reduzindo a expansão de colunas. | Boa alternativa para variáveis nominais com muitas categorias. |

### 6. Agrupamento em Classes (*Binning / Discretization*)

* **Conceito:** Transformar dados contínuos em intervalos discretos (ex.: idades divididas em faixas: `[18-30]`, `[31-50]`, `[51+]`).
* **Compromisso (*Trade-off*):** Aumenta a robustez contra o ruído e reduz o sobreajuste, mas **sacrifica a resolução e o detalhe matemático da informação**.

### 7. Engenharia de Atributos (*Feature Engineering*)

> *«O processo criativo de derivar novas variáveis com base nas já existentes para aumentar o poder preditivo do modelo.»*

* **Caso Prático (Timestamp: `2021-10-29 16h30`):**  
  Um modelo não compreende datas brutas de forma linear. A engenharia de variáveis decompõe-nas em múltiplos sinais explicativos:
  * Ano, Mês, Dia do Mês;
  * Hora e Minuto do dia (para captar períodos de ponta);
  * Dia da semana (ex.: 5ª Feira);
  * *Flags* booleanas calculadas: `É_Fim_De_Semana?` (Sim/Não), `É_Feriado?` (Sim/Não).


# Prática

## 1. Algoritmos: Árvores de Decisão (*Decision Trees*)

Uma Árvore de Decisão é um modelo preditivo estruturado sob a forma de um grafo hierárquico:
* **Nós internos / Ramos:** Representam testes sobre atributos (*features*).
* **Folhas (*Leaf nodes*):** Representam o resultado final predito.

```
                  [ Rendimento > 30k€? ]
                        /        \
                     Sim          Não
                     /              \
            [ Idade > 25? ]        [ Risco: ALTO ] (Folha)
               /       \
            Sim         Não
            /             \
    [ Risco: BAIXO ]    [ Risco: MÉDIO ] (Folhas)
```

### Classificação vs. Regressão
* **DT Classifier (Alvo Discreto / Classes):**
  * *Binário:* Sobrevivência no Titanic ($0$ ou $1$), Aprovação de crédito (Sim/Não).
  * *Multiclasse:* Identificação de tipos de fruta (*Maçã*, *Laranja*, *Pera*).
* **DT Regressor (Alvo Contínuo / Numérico):**
  * Estimativa da tarifa paga no Titanic em € (*Fare*), velocidade do tráfego em km/h.

> **Atenção Técnica no Scikit-Learn:**  
> As implementações `DecisionTreeClassifier` e `DecisionTreeRegressor` do Scikit-Learn **não aceitam variáveis categóricas em formato texto (*strings*)** diretamente. É obrigatório convertê-las previamente (via *One-Hot Encoding* ou *Label Encoding*) ou removê-las antes de executar o `.fit()`.

---

## 2. Métricas de Avaliação para Classificação

A escolha da métrica depende do equilíbrio entre classes e do impacto dos diferentes tipos de erro.

### A. Matriz de Confusão

Cruza a classe real (*Ground Truth*) com a classe prevista pelo modelo:

| | **Real: Positivo (1)** | **Real: Negativo (0)** |
| :--- | :---: | :---: |
| **Previsto: Positivo (1)** | **TP** (*True Positive*) | **FP** (*False Positive*) — **Erro Tipo I** |
| **Previsto: Negativo (0)** | **FN** (*False Negative*) — **Erro Tipo II** | **TN** (*True Negative*) |

* **Erro Tipo I (Falso Positivo):** Prever que o evento ocorreu quando não ocorreu (ex.: teste acusa gravidez num homem).
* **Erro Tipo II (Falso Negativo):** Não detetar o evento real (ex.: teste negativo numa grávida em trabalho de parto). Em medicina e deteção de fraude, este é habitualmente o erro mais grave.

---

### B. Métricas Derivadas

#### 1. Exatidão (*Accuracy*)
Proporção global de previsões corretas:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* *Limitação:* Inadequada para classes desbalanceadas (ex.: detetar doença rara que afeta 1% dos doentes).

#### 2. Precisão (*Precision*)
De todas as instâncias que o modelo previu como positivas, quantas são de facto positivas?
$$\text{Precision} = \frac{TP}{TP + FP}$$
* *Foco:* Minimizar Falsos Positivos (ex.: filtros de spam).

#### 3. Revocação / Sensibilidade (*Recall / Sensitivity*)
De todas as instâncias que eram verdadeiramente positivas, quantas conseguiu o modelo capturar?
$$\text{Recall} = \frac{TP}{TP + FN}$$
* *Foco:* Minimizar Falsos Negativos (ex.: diagnóstico oncológico, deteção de fraude).

#### 4. $F_1\text{-Score}$ e $F_\beta\text{-Score}$
* **$F_1\text{-Score}$:** Média harmónica entre Precisão e Revocação, dando peso idêntico a ambas:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **$F_\beta\text{-Score}$:** Permite ajustar a importância relativa entre Revocação e Precisão:
  $$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$
  * $\beta = 1 \implies F_1\text{-Score}$
  * $\beta > 1$ (ex.: $\beta=2$): Dá maior peso ao **Recall** (tolerância zero a Falsos Negativos).
  * $\beta < 1$ (ex.: $\beta=0.5$): Dá maior peso à **Precisão** (tolerância zero a Falsos Positivos).

---

### C. Curva ROC e AUC (*Area Under Curve*)

* **Curva ROC:** Gráfico que mapeia o **TPR** (*True Positive Rate* = Recall) face ao **FPR** (*False Positive Rate* = $\frac{FP}{FP + TN}$) para todos os possíveis limiares de decisão (*thresholds*).
* **AUC:** Mede a capacidade global do modelo em ordenar as observações positivas acima das negativas:
  * $\text{AUC} = 1.0$: Modelo perfeito.
  * $\text{AUC} = 0.5$: Desempenho equivalente ao acaso (linha diagonal de 45°).
  * $\text{AUC} < 0.5$: Pior do que o acaso (previsões invertidas).

---

## 3. Métricas de Avaliação para Regressão

Medem o desvio entre os valores reais ($y_j$) e os valores previstos ($\hat{y}_j$) para $n$ observações. Em todas elas, **quanto menor o valor, melhor é o modelo**.

| Métrica | Expressão Matemática | Características & Interpretação |
| :--- | :---: | :--- |
| **MAE** <br>(*Mean Absolute Error*) | $\text{MAE} = \frac{1}{n} \sum_{j=1}^{n} \|y_j - \hat{y}_j\|$ | **Linear:** Erro médio absoluto. Expressa o erro na **mesma unidade** da variável alvo. Menos sensível a erros extremos (*outliers*). |
| **MSE** <br>(*Mean Squared Error*) | $\text{MSE} = \frac{1}{n} \sum_{j=1}^{n} (y_j - \hat{y}_j)^2$ | **Quadrática:** Eleva as diferenças ao quadrado. **Penaliza fortemente erros de grande magnitude**. Unidade ao quadrado da variável original. |
| **RMSE** <br>(*Root Mean Squared Error*) | $\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (y_j - \hat{y}_j)^2}$ | **Equilibrada:** Mantém a forte penalização sobre erros grandes imposta pelo quadrado, mas devolve o resultado à **unidade original** da variável ($€, \text{km/h}, \dots$). |

### Exemplo Prático: Sensibilidade a Erros Graves
Analise o comportamento perante dois cenários com 4 previsões:

$$\begin{array}{l|c|c|c}
\textbf{Cenário} & \textbf{MAE} & \textbf{MSE} & \textbf{RMSE} \\ \hline
\text{A: Erros moderados e regulares } [1, -1, 3, 3] & 2.0 & 5.0 & 2.24 \\
\text{B: Três acertos perfeitos e um erro grave } [0, 0, 0, 10] & 2.5 & \mathbf{25.0} & \mathbf{5.00}
\end{array}$$

*Conclusão:* Enquanto o MAE quase não variou ($2.0 \to 2.5$), o MSE e o RMSE dispararam no Cenário B, provando serem ideais quando erros isolados de grande dimensão são intoleráveis.

---

## 4. Metodologias de Validação de Modelos

Avaliar o modelo nos mesmos dados em que foi treinado induz falsos sucessos (*overfitting*). É obrigatório testar com dados nunca vistos.

### 1. Hold-out Validation
* **Funcionamento:** Partição única dos dados em Treino e Teste (ex.: $80/20$, $75/25$).
* **Vantagem:** Muito rápido e computacionalmente simples.
* **Problema:** Alta variância no resultado (a métrica obtida depende do corte aleatório efetuado; pode haver amostras sub-representadas num dos lados).

---

### 2. $k$-Fold Cross Validation
* **Funcionamento:**
  1. O dataset é dividido em $k$ blocos (*folds*) de igual dimensão.
  2. São realizadas $k$ iterações. Em cada uma, utilizam-se **$k-1$ blocos para treino** e o **bloco restante para teste**.
  3. O erro final reportado é a média aritmética dos $k$ testes:
     $$E = \frac{1}{k}\sum_{i=1}^{k} E_i$$
* **Regra prática:** Habitualmente fixa-se **$k = 10$**.
* **Propriedades:** Mais dobras (*folds*) produzem estimativas com **menor viés** (*bias*), mas acarretam maior custo computacional.

---

### 3. Leave-One-Out Cross Validation (LOOCV)
* **Funcionamento:** Caso limite em que **$k = N$** ($N$ = número total de instâncias).
* Cada amostra individual é usada como conjunto de teste de forma isolada, enquanto as restantes $N-1$ servem para treino.
* **Aplicação ideal:** Conjuntos de dados **muito pequenos** (garante o máximo possível de dados para treino). Inviável em conjuntos de dados grandes devido ao tempo computacional.

---

## 5. Resumo de Implementação em Python (Scikit-Learn)

### Pipeline Completa de Validação e Avaliação

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, root_mean_squared_error
)

# 1. Preparação (X e y)
X = df.drop(columns=['Target_Class', 'colunas_texto_nao_codificadas'])
y = df['Target_Class']

# 2. Divisão Hold-Out (75% treino, 25% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2021)

# 3. Treino e Previsão
clf = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=2021)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 4. Avaliação de Classificação
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# 5. Validação Cruzada k-Fold Direta (k=10)
scores = cross_val_score(clf, X, y, cv=10)
print(f"Precisão k-Fold: {scores.mean():.2f} (+/- {scores.std():.2f})")
```


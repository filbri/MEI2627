#MEI-DAA #python #ML 

Se o NumPy são os tijolos, o Pandas é a argamassa e o Matplotlib/Seaborn são os olhos, o **Scikit-Learn é a fábrica de inteligência**. É ele que contém os algoritmos preditivos, os métodos de validação rigorosos e os transformadores de dados.

A grande genialidade do Scikit-Learn é a sua **consistência**. Todos os modelos do mundo (uma regressão linear simples ou uma floresta aleatória complexa) usam rigorosamente a mesma interface de três passos: **instanciar**, **`fit()`** e **`predict()`**.

# 🤖 Manual Prático & Intuitivo de Scikit-Learn para ML

```python
import numpy as np
import pandas as pd
import sklearn
```

## Módulo 1: O Mantra da API (`fit`, `transform` e `predict`)

O Scikit-Learn divide as suas ferramentas em duas grandes famílias. Se decorares esta regra, nunca mais terás erros de código:

```
1. Transformadores (Limpam/Escalam dados):
   - fit(): Aprende os parâmetros estatísticos (ex: média e desvio padrão dos dados de treino).
   - transform(): Aplica a fórmula matemática aos dados.
   - fit_transform(): Atalho conveniente (aprende e transforma logo a seguir).

2. Estimadores / Modelos (Fazem previsões):
   - fit(): Treina o algoritmo com base em X e y.
   - predict(): Cospe as previsões para novas amostras.
```

### ⚠️ A Regra de Ouro contra o *Data Leakage* (Vazamento de Dados):
> **Nunca uses `fit()` ou `fit_transform()` nos dados de teste!** O modelo tem de aprender as estatísticas **apenas** no conjunto de treino. Nos dados de teste, aplicas estritamente o `.transform()`.

## Módulo 2: Divisão de Dados (`train_test_split`)

Antes de tocar em qualquer algoritmo, tens de esconder uma parte dos teus dados para simular o "mundo real" e avaliar o modelo honestamente.

```python
from sklearn.model_selection import train_test_split

# Criando um dataset de exemplo com Pandas
df = pd.DataFrame({
    'Idade': [22, 45, 30, 52, 40, 25, 60, 35],
    'Salario': [1200, 3500, 2100, 4800, 3100, 1500, 5500, 2600],
    'Comprou': [0, 1, 0, 1, 1, 0, 1, 0] # Target (y)
})

X = df.drop(columns=['Comprou']) # Matriz de Features 2D
y = df['Comprou']                # Vetor Alvo 1D

# test_size=0.25 -> 25% para teste, 75% para treino
# random_state=42 -> Garante que a divisão é reprodutível
# stratify=y -> OBRIGATÓRIO em classificação! Mantém a mesma proporção de 0s e 1s em ambos os lados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")
```

## Módulo 3: Normalização e Escala (`StandardScaler` vs `MinMaxScaler`)

Algoritmos sensíveis a distâncias ou gradientes (Regressão Logística, SVM, KNN, Redes Neurais) são enganados se uma variável tiver valores entre $0$ e $1$ e outra entre $10.000$ e $100.000$.

* **`StandardScaler`:** Transforma os dados para Média = 0 e Desvio Padrão = 1 (Z-score). O mais usado!
* **`MinMaxScaler`:** Comprime tudo entre 0 e 1.
* **`RobustScaler`:** Usa a mediana e o IQR, sendo ideal se existirem muitos *outliers*.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 1. Aprende estatísticas E transforma o Treino:
X_train_scaled = scaler.fit_transform(X_train)

# 2. Apenas transforma o Teste com o que aprendeu no treino:
X_test_scaled = scaler.transform(X_test)
```
> 💡 *Modelos baseados em árvores de decisão (Decision Tree, Random Forest, XGBoost) são completamente imunes à escala das variáveis e não precisam de normalização!*

## Módulo 4: Tratamento de Nulos e Categóricos (`Imputer` e `OneHotEncoder`)

Em vez de limpares tudo manualmente no Pandas, o Scikit-Learn tem transformadores reproduzíveis para o ambiente de produção:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# A. Tratar Valores em Falta (NaN)
imputer = SimpleImputer(strategy='median') # ou 'mean', 'most_frequent'
# X_imputado = imputer.fit_transform(X_com_nulos)

# B. One-Hot Encoding (Variáveis Categóricas em 0 e 1)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# handle_unknown='ignore': Se aparecer uma categoria nova no teste, preenche com zeros sem dar erro!
cidades = np.array([['Lisboa'], ['Porto'], ['Coimbra'], ['Lisboa']])
cidades_codificadas = encoder.fit_transform(cidades)

print(cidades_codificadas)
```

## Módulo 5: Regressão Linear e Regularização (Ridge & Lasso)

Para prever números contínuos (preço de casas, vendas futuras, temperatura):

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. Instanciar
modelo_reg = Ridge(alpha=1.0) # Ridge adiciona regularização L2 contra sobreajuste

# 2. Treinar
modelo_reg.fit(X_train_scaled, y_train)

# 3. Prever no conjunto de teste
y_pred = modelo_reg.predict(X_test_scaled)

# 4. Avaliar
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente R² (0 a 1): {r2:.2f}")

# Inspecionar os pesos aprendidos:
print("Pesos das Features:", modelo_reg.coef_)
print("Interceto (Bias):", modelo_reg.intercept_)
```

## Módulo 6: Classificação (`LogisticRegression` e `RandomForest`)

Para prever categorias (Fraude ou Não Fraude, Gato ou Cão):

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Vamos usar uma Floresta Aleatória com 100 árvores
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Treinar
clf.fit(X_train, y_train) # Não precisa de escala prévia!

# 1. Previsão de Classes (0 ou 1)
y_pred_classe = clf.predict(X_test)

# 2. Previsão de Probabilidades (Crucial para escolher limiares de decisão)
y_pred_prob = clf.predict_proba(X_test)
print("Probabilidade [P(Classe 0), P(Classe 1)] da 1ª amostra:\n", y_pred_prob[0])
```

## Módulo 7: Avaliação Rigorosa de Classificação (Para além da Acurácia)

Em datasets desbalanceados (ex: 99% não-fraude e 1% fraude), um modelo burro que diga sempre "não-fraude" terá 99% de acurácia. **Nunca confies apenas na acurácia!**

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Relatório Completo: Precision, Recall e F1-Score por classe
print(classification_report(y_test, y_pred_classe))

# Matriz de Confusão
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_classe))

# ROC-AUC (Avalia a capacidade do modelo em separar as classes de 0.5 a 1.0)
# Usa sempre as probabilidades da classe positiva: y_pred_prob[:, 1]
auc = roc_auc_score(y_test, y_pred_prob[:, 1])
print(f"Área sob a Curva ROC (AUC): {auc:.3f}")
```

## Módulo 8: Validação Cruzada (*K-Fold Cross-Validation*)

Testar apenas numa única divisão de teste pode ser enganador por mero acaso do sorteio. O K-Fold divide os teus dados de treino em $K$ partes (geralmente 5) e testa o modelo 5 vezes em rotação:

```
K=5 Folds:
Fold 1: [ Teste | Treino | Treino | Treino | Treino ] -> Score 1
Fold 2: [ Treino | Teste | Treino | Treino | Treino ] -> Score 2
Fold 3: [ Treino | Treino | Teste | Treino | Treino ] -> Score 3
...
```

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# StratifiedKFold preserva a proporção das classes em cada fold
cv_estrategia = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Avaliar o modelo usando F1-Score como métrica
scores = cross_val_score(clf, X, y, cv=cv_estrategia, scoring='f1')

print("F1-Scores em cada dobra:", scores)
print(f"F1 Médio: {scores.mean():.3f} (± {scores.std():.3f})")
```

## Módulo 9: O Santo Graal do ML: `Pipeline` e `ColumnTransformer`

Em código amador, as pessoas limpam os dados fora do modelo, causando erros e vazamento de informação. 

Os **Pipelines** juntam todo o fluxo (imputação + escalamento + codificação + modelo) num único bloco fechado à prova de falhas:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Separar as colunas por tipo
colunas_num = ['Idade', 'Salario']
colunas_cat = ['Cidade'] if 'Cidade' in df.columns else []

# 2. Pipeline para dados numéricos
pipe_num = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 3. Pipeline para dados categóricos
pipe_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 4. O Maestro: ColumnTransformer que aplica cada pipeline às respetivas colunas
preprocessor = ColumnTransformer(transformers=[
    ('num', pipe_num, colunas_num),
    ('cat', pipe_cat, colunas_cat)
])

# 5. O Pipeline Final Completo (Transformação + Modelo)
pipeline_final = Pipeline([
    ('pre', preprocessor),
    ('modelo', LogisticRegression())
])

# ✅ COM UM ÚNICO COMANDO TREINAS O PROCESSO INTEIRO:
pipeline_final.fit(X_train, y_train)

# ✅ COM UM ÚNICO COMANDO APLICAS TUDO AO TESTE E PREVÊS:
previsoes = pipeline_final.predict(X_test)
```

## Módulo 10: Otimização de Hiperparâmetros (`GridSearchCV`)

Não sabes qual a melhor profundidade para a tua árvore de decisão ou o melhor valor de regularização? Deixa o Scikit-Learn testar todas as combinações com validação cruzada:

```python
from sklearn.model_selection import GridSearchCV

# Definir a grelha de combinações que queremos testar
grade_parametros = {
    'modelo__C': [0.01, 0.1, 1.0, 10.0],       # Parâmetro da LogisticRegression
    'modelo__penalty': ['l2']
}

# GridSearchCV testa cada combinação com 5-Fold Cross-Validation
grid_search = GridSearchCV(
    estimator=pipeline_final,
    param_grid=grade_parametros,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1 # Usa todos os núcleos do processador para acelerar!
)

# Executar a busca
grid_search.fit(X_train, y_train)

print("Melhores Hiperparâmetros:", grid_search.best_params_)
print(f"Melhor ROC-AUC Médio no Treino: {grid_search.best_score_:.3f}")

# O melhor modelo já vem pronto a usar:
melhor_modelo = grid_search.best_estimator_
resultado_teste = melhor_modelo.score(X_test, y_test)
```


## 📋 Folha de Consulta Rápida de Scikit-Learn (Cheat Sheet)

| Desafio no Fluxo de ML | Ferramenta do Scikit-Learn |
| :--- | :--- |
| **Separar Treino e Teste** | `train_test_split(X, y, test_size=0.2, stratify=y)` |
| **Padronizar Variáveis (Z-Score)** | `StandardScaler()` (`fit_transform` / `transform`) |
| **Substituir Nulos** | `SimpleImputer(strategy='median')` |
| **Converter Texto Categórico** | `OneHotEncoder(handle_unknown='ignore')` |
| **Prever Classes (Classificação)** | `model.predict(X)` |
| **Prever Probabilidades** | `model.predict_proba(X)` |
| **Métricas de Classificação** | `classification_report(y_true, y_pred)` |
| **Métricas de Regressão** | `root_mean_squared_error(y_true, y_pred)` |
| **Validação em Vários Folds** | `cross_val_score(model, X, y, cv=5, scoring='...')` |
| **Juntar Pré-processamento e Modelo** | `Pipeline([('prep', ...), ('model', ...)])` |
| **Afinar Hiperparâmetros** | `GridSearchCV(pipeline, param_grid, cv=5)` |


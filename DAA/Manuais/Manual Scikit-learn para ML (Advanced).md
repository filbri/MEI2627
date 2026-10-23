#MEI-DAA #python #ML 

## Módulo 11: Otimização Rápida com `RandomizedSearchCV`

O `GridSearchCV` (visto no Módulo 10) testa *todas* as combinações possíveis. Se tiveres 5 parâmetros com 4 opções cada, são centenas de treinos — o que pode demorar horas ou dias.

O **`RandomizedSearchCV`** é a escolha dos profissionais: em vez de testar tudo, sorteia um número fixo de combinações aleatórias (`n_iter`), cobrindo uma área muito maior do espaço de busca em muito menos tempo.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Definir distribuições contínuas/discretas em vez de listas fixas
distribuicao_parametros = {
    'n_estimators': randint(50, 300),      # Sorteia inteiros entre 50 e 300
    'max_depth': randint(3, 15),           # Sorteia profundidades entre 3 e 15
    'min_samples_split': [2, 5, 10]
}

clf = RandomForestClassifier(random_state=42)

# n_iter=20: Testa apenas 20 combinações aleatórias em vez de centenas!
busca_rapida = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distribuicao_parametros,
    n_iter=20,
    cv=3,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

# busca_rapida.fit(X_train, y_train)
# print("Melhores parâmetros encontrados:", busca_rapida.best_params_)
```

## Módulo 12: Redução de Dimensionalidade com PCA (`PCA`)

Quando o teu dataset tem 50 ou 100 *features*, muitas delas são ruído ou redundancy. A **Análise de Componentes Principais (PCA)** comprime essas variáveis em eixos matemáticos que retêm o máximo de informação (variância) possível.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# O PCA exige SEMPRE que os dados estejam padronizados primeiro!
X_escalado = StandardScaler().fit_transform(X_train)

# n_components=0.95 -> Mantém quantos componentes forem precisos para explicar 95% da variância!
pca = PCA(n_components=0.95, random_state=42)
X_reduzido = pca.fit_transform(X_escalado)

print(f"Colunas originais: {X_train.shape[1]}")
print(f"Colunas após PCA: {X_reduzido.shape[1]}")
print("Variância explicada por cada componente:", pca.explained_variance_ratio_)
```
> 💡 *Se precisares de visualizar dados multidimensionais num gráfico 2D do Matplotlib, usa `PCA(n_components=2)`!*

## Módulo 13: Agrupamento com K-Means (`KMeans`)

Nem todos os problemas têm rótulos conhecidos ($y$). O **K-Means** é o algoritmo de *Machine Learning Não Supervisionado* mais utilizado para encontrar segmentos naturais de clientes ou produtos.

```python
from sklearn.cluster import KMeans

# Queremos agrupar os nossos dados em 3 grupos (clusters)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)

# Como não há y, fazemos fit_predict apenas em X!
clusters_atribuidos = kmeans.fit_predict(X_escalado)

print("Cluster atribuído a cada amostra:", clusters_atribuidos[:10])

# Coordenadas do centro de gravidade de cada cluster:
centroides = kmeans.cluster_centers_

# Inércia (Soma das distâncias quadráticas ao quadrado — usada para o método do Cotovelo/Elbow):
print("Inércia do agrupamento:", kmeans.inertia_)
```

## Módulo 14: Avaliação Visual Nativa do Sklearn (`*Display`)

O Scikit-Learn moderno inclui classes de visualização nativas (`ConfusionMatrixDisplay`, `RocCurveDisplay`, etc.) que eliminam a necessidade de escrever dezenas de linhas de Matplotlib à mão:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 1. Matriz de Confusão elegante desenhada diretamente do modelo e do teste
ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, 
    cmap='Blues', 
    display_labels=['Não', 'Sim'], 
    ax=axes[0]
)
axes[0].set_title("Matriz de Confusão")

# 2. Curva ROC com cálculo automático de AUC
RocCurveDisplay.from_estimator(
    clf, X_test, y_test, 
    ax=axes[1], 
    color='darkorange'
)
axes[1].plot([0, 1], [0, 1], 'k--', label='Baseline')
axes[1].set_title("Curva ROC Nativa")

plt.tight_layout()
plt.show()
```

## Módulo 15: Lidar com Classes Desbalanceadas (`class_weight='balanced'`)

Se estás a tentar prever Fraude bancária (onde 99.8% das transações são legítimas e 0.2% são fraude), os algoritmos convencionais ignoram a classe minoritária para minimizar o erro global.

A forma mais simples e robusta no Scikit-Learn de contrariar isto sem recorrer a técnicas externas de amostragem é ativar o **`class_weight='balanced'`**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# O Sklearn ajusta automaticamente os pesos das classes de forma inversamente
# proporcional às suas frequências nos dados de treino!
modelo_justo = LogisticRegression(class_weight='balanced', max_iter=1000)
floresta_justa = RandomForestClassifier(class_weight='balanced', random_state=42)

# Ambos penalizam fortemente os erros cometidos na classe rara.
```

## Módulo 16: Guardar e Carregar Modelos para Produção (`joblib`)

Depois de horas a ajustar hiperparâmetros, não podes perder o teu modelo quando fechas o JupyterLab. O **`joblib`** é o padrão recomendado pelo Scikit-Learn para serializar estimadores pesados e pipelines inteiros:

```python
import joblib

# 1. GUARDAR o modelo ou o Pipeline final num ficheiro binário
joblib.dump(pipeline_final, 'modelo_churn_v1.joblib')
print("Modelo guardado com sucesso!")

# -------------------------------------------------------------
# 2. CARREGAR noutra aplicação (ex: numa API com FastAPI ou Flask)
modelo_em_producao = joblib.load('modelo_churn_v1.joblib')

# Fazer predições instantâneas em novos dados sem precisar de treinar nada:
novas_amostras = pd.DataFrame({'Idade': [29], 'Salario': [3200]})
previsao = modelo_em_producao.predict(novas_amostras)
print("Previsão em produção:", previsao)
```

## Módulo 17: Transformações Personalizadas em Pipelines (`FunctionTransformer`)

E se precisares de aplicar uma fórmula matemática específica (como `np.log1p`) a uma coluna dentro do teu `Pipeline`, garantindo que não há vazamento de dados?

Usa o **`FunctionTransformer`**:

```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

# Criar um transformador a partir de qualquer função Python existente
transformador_log = FunctionTransformer(np.log1p, validate=True)

# Integrar perfeitamente num pipeline
pipe_com_log = make_pipeline(
    transformador_log,
    StandardScaler(),
    LogisticRegression()
)
```

## Módulo 18: Interpretação Visual de Árvores de Decisão (`plot_tree`)

Quando um gestor ou auditor te perguntar *"Porque é que o modelo rejeitou este empréstimo?"*, podes desenhar o fluxo lógico exato de uma Árvore de Decisão:

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Treinar uma árvore pequena para fácil interpretação
arvore = DecisionTreeClassifier(max_depth=3, random_state=42)
arvore.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(12, 6))

plot_tree(
    arvore, 
    feature_names=X_train.columns, 
    class_names=['Não', 'Sim'], 
    filled=True,          # Colore os nós conforme a classe dominante
    rounded=True, 
    fontsize=9,
    ax=ax
)

plt.title("Regras de Decisão da Árvore", fontsize=14)
plt.show()
```

## 📌 Tabela de Consulta Expandida (Cheat Sheet Completa)

| Desafio Prático de ML | Ferramenta do Scikit-Learn |
| :--- | :--- |
| **Busca de Hiperparâmetros Rápida** | `RandomizedSearchCV(model, params, n_iter=20)` |
| **Redução de Dimensionalidade** | `PCA(n_components=0.95)` |
| **Segmentação / Clusters Não Supervisionados** | `KMeans(n_clusters=k)` |
| **Desenhar Matriz de Confusão em 1 linha** | `ConfusionMatrixDisplay.from_estimator(model, X, y)` |
| **Desenhar Curva ROC em 1 linha** | `RocCurveDisplay.from_estimator(model, X, y)` |
| **Lidar com Desbalanceamento Severo** | Definir `class_weight='balanced'` no modelo |
| **Exportar Modelo para Ficheiro** | `joblib.dump(pipeline, 'modelo.joblib')` |
| **Importar Modelo em Produção** | `joblib.load('modelo.joblib')` |
| **Transformar Funções em Passos do Pipeline** | `FunctionTransformer(np.log1p)` |
| **Visualizar Nós e Regras da Árvore** | `plot_tree(arvore, feature_names=..., filled=True)` |


---

# 🚀 Continuação do Manual: As Operações Mais Usadas em NumPy

---
#python #ML #MEI-DAA

## 11. Juntar e Empilhar Arrays (*Stacking* e *Concatenation*)

Em ML, é muito comum teres conjuntos de *features* vindos de fontes diferentes e precisares de os unir numa única matriz $X$, ou adicionar uma coluna de $1$s para o termo de *bias* (interceto).

### A. `np.concatenate`
Junta arrays ao longo de um eixo existente.

```python
# Duas matrizes de features (mesmo número de amostras, colunas diferentes)
X_numericas = np.array([[23, 50000], 
                        [45, 80000]]) # shape (2, 2)

X_binarias  = np.array([[1], 
                        [0]])         # shape (2, 1)

# Juntar lado a lado (colunas) -> axis=1
X_completo = np.concatenate([X_numericas, X_binarias], axis=1)
print(X_completo)
# [[2.3e+01 5.0e+04 1.0e+00]
#  [4.5e+01 8.0e+04 0.0e+00]]
```

### B. `np.vstack`, `np.hstack` e `np.column_stack` (Os atalhos intuitivos)
* **`np.vstack` (Vertical):** Empilha linhas (adiciona mais amostras/dados).
* **`np.hstack` (Horizontal):** Empilha colunas (adiciona mais variáveis).
* **`np.column_stack`:** Transforma vetores 1D em colunas e junta-os de forma automática.

```python
# 💡 Truque clássico de ML: Adicionar a coluna de Bias (1s) para Regressão Linear
X = np.array([[2.5], [3.0], [4.5]])
bias = np.ones((X.shape[0], 1))

# Matriz pronta para a Equação Normal [1, x]:
X_com_bias = np.hstack([bias, X])
print(X_com_bias)
```


## 12. Transformações Matemáticas Elemento a Elemento (*Ufuncs*)

Em pré-processamento de dados e cálculo de métricas de avaliação, usamos funções matemáticas vetorizadas.

```python
x = np.array([1.0, 10.0, 100.0])

# 1. Escalas Logarítmicas (Crucial para dados com cauda longa / skewed)
# np.log1p(x) calcula log(1 + x), evitando o erro de log(0) se houver zeros!
log_x = np.log1p(x)

# 2. Exponencial (base da função Sigmoide e Softmax)
exp_x = np.exp(x)

# 3. Raiz quadrada e valor absoluto (Cálculo manual de RMSE e MAE)
erros = np.array([-2.5, 0.0, 3.2])
mae = np.mean(np.abs(erros))                 # Mean Absolute Error
rmse = np.sqrt(np.mean(erros ** 2))         # Root Mean Squared Error
```

### 💡 Caso Real: Implementar a função Sigmoide em 1 linha
A função de ativação da Regressão Logística e de Redes Neurais:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-2.0, 0.0, 2.0])
print(sigmoid(z))  # [0.1192, 0.5, 0.8807]
```


## 13. Ordenação e os Mágicos "Args" (`np.argsort`)

Ordenar valores é simples (`np.sort`), mas em Machine Learning o que mais queremos são os **índices** ordenados (`np.argsort`). 

É com `argsort` que se implementa, por exemplo, o algoritmo **KNN (K-Nearest Neighbors)** ou se descobre o **Top-K previsões**.

```python
# Distâncias calculadas de uma nova amostra para 5 amostras conhecidas:
distancias = np.array([12.4, 3.1, 45.0, 1.2, 8.7])

# np.sort() devolve os VALORES ordenados:
print(np.sort(distancias))  # [ 1.2,  3.1,  8.7, 12.4, 45. ]

# np.argsort() devolve os ÍNDICES que ordenariam o array:
indices_ordenados = np.argsort(distancias)
print(indices_ordenados)    # [3, 1, 4, 0, 2]

# Para pegar nos 3 vizinhos mais próximos (K=3):
k_vizinhos = indices_ordenados[:3]
print("Vizinhos mais próximos:", k_vizinhos)  # Amostra 3, 1 e 4
```


## 14. Limpeza e Proteção: `np.clip`, `np.isnan` e Funções `nan*`

Na prática da ciência de dados, os dados chegam sujos, com valores nulos (`NaN`) e valores absurdamente altos.

### A. `np.clip` (Corte de Limites)
Evita explosão de gradientes ou valores proibidos (como probabilidades $0$ ou $1$ em *Log Loss* que causariam $\log(0) = -\infty$).

```python
probabilidades = np.array([0.0, 0.45, 0.99, 1.0])

# Força todos os valores a ficarem estritamente entre 1e-15 e (1 - 1e-15)
prob_seguras = np.clip(probabilidades, 1e-15, 1 - 1e-15)
print(prob_seguras)
```

### B. Lidar com `NaN` (*Not a Number*)
```python
dados_com_nulos = np.array([10.0, np.nan, 20.0, 30.0])

# 1. Identificar nulos (retorna máscara booleana)
print(np.isnan(dados_com_nulos))  # [False,  True, False, False]

# 2. Contar quantos nulos existem
total_nulos = np.isnan(dados_com_nulos).sum()  # 1

# 3. Estatísticas conscientes de NaNs (as funções com prefixo 'nan')
print(np.mean(dados_com_nulos))     # Retorna NaN (inútil!)
print(np.nanmean(dados_com_nulos)) # 20.0 (ignora o valor nulo!)
print(np.nanstd(dados_com_nulos))  # Desvio padrão ignorando o NaN
```


## 15. Contagem e Desbalanceamento: `np.unique`

Antes de treinar um classificador, precisas de saber se as tuas classes estão equilibradas (ex: 50% Fraude vs 50% Não-Fraude).

```python
classes = np.array([0, 1, 0, 0, 0, 1, 0, 2, 0])

# Obter valores únicos e as respetivas contagens
valores, contagens = np.unique(classes, return_counts=True)

for val, count in zip(valores, contagens):
    print(f"Classe {val}: {count} ocorrências ({count / len(classes):.1%})")
# Classe 0: 6 ocorrências (66.7%) -> Dataset desbalanceado!
```


## 16. Distâncias e Normas Vetoriais (`np.linalg.norm`)

A distância euclidiana é o motor de algoritmos como K-Means, KNN e de funções de custo com regularização L1/L2 (Lasso / Ridge).

```python
# Dois pontos no espaço de features (ex: peso e altura)
ponto_a = np.array([70.0, 1.75])
ponto_b = np.array([85.0, 1.82])

# Distância Euclidiana (Norma L2)
dist_euclidiana = np.linalg.norm(ponto_a - ponto_b)
print("Distância L2:", dist_euclidiana)

# Distância de Manhattan (Norma L1)
dist_manhattan = np.linalg.norm(ponto_a - ponto_b, ord=1)
print("Distância L1:", dist_manhattan)
```


## 17. Verificações Lógicas e Acumulação (`any`, `all`, `cumsum`)

### A. `np.any` e `np.all` (Validações rápidas)
```python
pesos = np.array([0.5, 1.2, np.nan, -0.4])

# Existe algum NaN no modelo?
tem_nan = np.any(np.isnan(pesos))  # True

# Todos os pesos são positivos?
todos_positivos = np.all(pesos > 0)  # False
```

### B. `np.cumsum` (Soma Acumulada - Essencial para PCA)
Na Análise de Componentes Principais (PCA), usamos a soma cumulativa para escolher quantos componentes explicam 95% da variância:

```python
# Variância explicada por cada componente principal (simulada)
variancia_explicada = np.array([0.40, 0.30, 0.15, 0.10, 0.05])

variancia_acumulada = np.cumsum(variancia_explicada)
print(variancia_acumulada)
# [0.40, 0.70, 0.85, 0.95, 1.00]
# Conclusão: com 4 componentes atingimos 95% da informação!
```


## 📌 Cheat Sheet Expandida

| Operação | O que faz no teu código de ML |
| :--- | :--- |
| `np.hstack([bias, X])` | Adiciona coluna de bias / combina features |
| `np.vstack([treino, teste])` | Junta lotes de amostras verticalmente |
| `np.log1p(X)` | Normaliza variáveis enviesadas sem dar erro com zeros |
| `np.argsort(dist)` | Devolve os índices dos menores/maiores (KNN/Top-K) |
| `np.clip(arr, min, max)` | Impede *exploding gradients* e erros de $\log(0)$ |
| `np.isnan(X).sum()` | Conta dados em falta na matriz |
| `np.nanmean(X, axis=0)` | Calcula a média ignorando os valores nulos |
| `np.unique(y, return_counts=True)` | Avalia o desbalanceamento das classes |
| `np.linalg.norm(a - b)` | Calcula a distância euclidiana entre amostras |
| `np.cumsum(var)` | Analisa variância retida no PCA |

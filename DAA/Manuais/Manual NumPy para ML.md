#python #ML #MEI-DAA

O **NumPy** (*Numerical Python*) não é apenas uma biblioteca; é o **coração de todo o ecossistema de Ciência de Dados e Machine Learning em Python**. O Pandas é construído sobre ele, o Scikit-Learn exige arrays NumPy por baixo dos panos, e ferramentas como PyTorch ou TensorFlow usam tensores que são primos diretos do array do NumPy.

# Manual Prático & Intuitivo de NumPy para Data Science & ML

## 1. A Filosofia: Porque não usamos listas normais do Python?

Uma lista Python comum guarda ponteiros para objetos espalhados pela memória RAM. Já o array NumPy (`ndarray`) aloca um **bloco contíguo de memória** com dados do **mesmo tipo** (*homogéneo*), delegando o cálculo pesado para rotinas ultra-otimizadas escritas em C e Fortran.

```python
import numpy as np  # Convenção universal
```

* **Vetorização:** Operações aplicadas ao array inteiro de uma só vez, sem precisares de escrever ciclos `for`.
* **Velocidade:** Pode ser de 50 a 300 vezes mais rápido do que listas nativas.

## 2. Anatomia do Array: Dimensões, Shapes e Dtypes

Antes de avançares, tens de saber inspecionar qualquer array. Em Data Science, 80% dos erros acontecem porque o formato (*shape*) não bate certo.

```python
dados = np.array([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])

print(dados.shape)  # (2, 3) -> 2 linhas (amostras), 3 colunas (features)
print(dados.ndim)   # 2 dimensões (matriz)
print(dados.size)   # 6 elementos no total
print(dados.dtype)  # float64
```

### 💡 Dica de ML: Tipos de Dados (`dtype`)
Em deep learning e modelos pesados, mudar de `float64` para `float32` corta o consumo de memória RAM pela **metade**:
```python
x = np.array([1, 2, 3], dtype=np.float32)
# Ou converter um array existente:
x = x.astype(np.int32)
```

## 3. Criação de Dados Sintéticos e Inicialização

Em ML, criamos arrays do zero frequentemente para inicializar pesos de redes neurais, criar *placeholders* ou gerar curvas de decisão.

```python
# 1. Zeros e Uns (Muito usados para máscaras ou inicialização de vetores)
zeros = np.zeros(shape=(3, 2))       # Matriz 3x2 de zeros
uns = np.ones(shape=(4,))            # Vetor 1D de 4 posições

# 2. Sequências com np.arange vs np.linspace
# np.arange(início, fim_exclusivo, passo)
seq1 = np.arange(0, 10, 2)          # [0, 2, 4, 6, 8]

# np.linspace(início, fim_inclusivo, total_de_pontos)
# 🌟 O favorito para gráficos no Matplotlib!
grid = np.linspace(0, 1, 5)         # [0.  , 0.25, 0.5 , 0.75, 1.  ]

# 3. Matriz Identidade (útil em Álgebra Linear / Regularização Ridge)
I = np.eye(3)                       # Matriz 3x3 com 1 na diagonal principal
```

## 4. Indexação e Fatiamento (*Slicing*): Separar $X$ e $y$

Esta é **a tarefa número 1** em Machine Learning: tens um conjunto de dados e queres separar as variáveis independentes ($X$) da variável dependente ($y$).

A sintaxe no NumPy 2D é sempre: `array[linhas, colunas]`.

```python
# Simulando um dataset com 4 amostras e 3 colunas (2 features + 1 target)
dataset = np.array([
    [25, 50000, 1],   # Amostra 0: [Idade, Salário, Comprou?]
    [30, 65000, 1],   # Amostra 1
    [19, 18000, 0],   # Amostra 2
    [45, 90000, 1]    # Amostra 3
])

# Separar X (todas as linhas, da coluna 0 até à penúltima)
X = dataset[:, :-1]
print("Features (X):\n", X)

# Separar y (todas as linhas, apenas a última coluna)
y = dataset[:, -1]
print("Target (y):\n", y)
```

### ⚠️ Cuidado: *Views* vs *Copies*
O fatiamento no NumPy cria apenas uma **visualização** (*view*). Se alterares a fatia, alteras os dados originais!
```python
sub = dataset[:2, :2]
sub[0, 0] = 999  # Altera também o 'dataset' original!

# Se quiseres isolar completamente:
sub_segura = dataset[:2, :2].copy()
```

## 5. Reshaping e o Truque do `-1` (Indispensável para Sklearn)

Muitas funções do Scikit-Learn (como o `StandardScaler` ou `LinearRegression`) recusam vetores unidimensionais de formato `(n,)`; elas exigem explicitamente matrizes 2D de formato `(n, 1)`.

```python
idades = np.array([22, 35, 48, 52])
print(idades.shape)  # (4,) -> Vetor 1D

# ❌ Isto daria erro em alguns métodos do Sklearn!
# ✅ A solução: .reshape(-1, 1)
# O '-1' diz: "NumPy, calcula tu o número de linhas necessário!"
idades_2d = idades.reshape(-1, 1)
print(idades_2d.shape)  # (4, 1) -> Matriz coluna 2D
```

### Manipulações úteis:
* **Transpor:** `X.T` (troca linhas por colunas).
* **Achatar para 1D:** `X.ravel()` (rápido, cria view) ou `X.flatten()` (cria cópia).
* **Adicionar eixo:** `np.expand_dims(x, axis=0)` (muito comum em redes neurais para criar a dimensão do *batch*).

## 6. O Mistério do `axis`: Decifrado de Vez!

Um dos conceitos que mais confunde quem está a começar é o parâmetro `axis` nas agregações (`sum`, `mean`, `std`, etc.).

Guarda esta regra de ouro: **O `axis` indica a dimensão que vai ser "comprimida" / eliminada.**

```
Matriz:
      Col 0   Col 1
L0 [   1,      2   ]
L1 [   3,      4   ]
```

* `axis=0` percorre **verticalmente** (colapsa as linhas $\downarrow$). Dá o resultado **por coluna** (média de cada feature).
* `axis=1` percorre **horizontalmente** (colapsa as colunas $\rightarrow$). Dá o resultado **por linha** (média de cada amostra).

```python
notas = np.array([
    [14, 16],  # Aluno A (Matemática, Física)
    [10, 12],  # Aluno B
    [18, 20]   # Aluno C
])

# Média de cada disciplina (ao longo das linhas):
media_disciplinas = notas.mean(axis=0)
print(media_disciplinas)  # [14. , 16.] -> shape (2,)

# Média de cada aluno (ao longo das colunas):
media_alunos = notas.mean(axis=1)
print(media_alunos)       # [15. , 11. , 19.] -> shape (3,)
```

## 7. Operações Vetorizadas e *Broadcasting*

Esquece os ciclos `for`. Se fizeres `a + b`, o NumPy soma elemento a elemento em microssegundos.

O **Broadcasting** é a capacidade mágica do NumPy de operar entre arrays com formatos diferentes, "esticando" virtualmente o menor para se ajustar ao maior:

```python
# Cenário Real de ML: Normalização Min-Max manual!
dados = np.array([
    [10.0, 200.0],
    [20.0, 400.0],
    [30.0, 600.0]
])

minimos = dados.min(axis=0)  # [10., 200.] (shape: 2,)
maximos = dados.max(axis=0)  # [30., 600.] (shape: 2,)

# O vetor 1D de tamanho 2 é "expandido" para subtrair de todas as 3 linhas!
dados_normalizados = (dados - minimos) / (maximos - minimos)
print(dados_normalizados)
# Resultado perfeitamente escalado entre 0 e 1!
```

## 8. Filtros e Máscaras Booleanas (Busca de Outliers)

Como encontrar e filtrar dados sem usar `if`? Usando máscaras booleanas.

```python
idades = np.array([15, 22, 45, 17, 60, 82, 12])

# 1. Cria a máscara (array de True/False)
maiores_idade = idades >= 18
print(maiores_idade)  # [False,  True,  True, False,  True,  True, False]

# 2. Aplica a máscara para extrair os valores
individuos_aptos = idades[maiores_idade]
print(individuos_aptos)  # [22, 45, 60, 82]

# 3. Múltiplas condições (Usa & para E, | para OU, e fecha sempre com parênteses!)
meia_idade = idades[(idades >= 40) & (idades <= 65)]
print(meia_idade)  # [45, 60]

# 4. np.where: O "SE/SENÃO" vetorizado
# Sintaxe: np.where(condição, valor_se_verdadeiro, valor_se_falso)
categorias = np.where(idades >= 18, "Adulto", "Menor")
print(categorias)
```

## 9. Dados Aleatórios com a Nova API (`default_rng`)

Desde o NumPy 1.17+, a forma moderna e recomendada de trabalhar com números pseudoaleatórios é através do `Generator` (o antigo `np.random.seed` já não é a melhor prática).

```python
# Criar o gerador com uma semente fixa (para reproduzibilidade de experiências de ML)
rng = np.random.default_rng(seed=42)

# Distribuição Normal / Gaussiana (média=0, desvio_padrão=1)
# Vital para gerar ruído ou inicializar pesos
pesos = rng.normal(loc=0.0, scale=1.0, size=(3, 2))

# Números inteiros aleatórios (ex: simular classes 0 ou 1)
labels = rng.integers(low=0, high=2, size=10)

# Baralhar dados (Shuffle de treino/teste)
indices = np.arange(len(labels))
rng.shuffle(indices)
```

## 10. Álgebra Linear na Prática de ML

Dois conceitos essenciais: produto escalar (*dot product*) e resolução de equações.

### A. Previsão de uma Regressão Linear: $\hat{y} = X \cdot w + b$
Usa o operador `@` (multiplicação de matrizes):

```python
# 3 amostras, 2 features
X = np.array([[1.0, 2.0], 
              [3.0, 4.0], 
              [5.0, 6.0]])

# 2 pesos (um por feature)
pesos = np.array([0.5, -0.2])
bias = 0.1

# Predição: Multiplicação Matricial (3x2) @ (2,) -> (3,)
y_pred = X @ pesos + bias
print("Previsões:", y_pred)
```

### B. `np.argmax`: A função de saída de Classificadores
Quando um modelo de Deep Learning cospe probabilidades para classes (ex: [Gato, Cão, Pássaro]), como sabemos qual foi a classe escolhida?

```python
# Probabilidades previstas para 3 amostras em 3 classes
probs = np.array([
    [0.1, 0.7, 0.2],  # Amostra 0 -> Classe 1 (Cão)
    [0.8, 0.1, 0.1],  # Amostra 1 -> Classe 0 (Gato)
    [0.05, 0.05, 0.9] # Amostra 2 -> Classe 2 (Pássaro)
])

classes_escolhidas = np.argmax(probs, axis=1)
print(classes_escolhidas)  # [1, 0, 2]
```

## 📋 Folha de Consulta Rápida (Cheat Sheet Mental)

| Se queres... | Usa isto: |
| :--- | :--- |
| Ver o formato das tuas matrizes | `arr.shape` |
| Ajustar formato para o Sklearn | `arr.reshape(-1, 1)` |
| Saber a média por coluna (por feature) | `arr.mean(axis=0)` |
| Saber a média por linha (por amostra) | `arr.mean(axis=1)` |
| Substituir valores condicionalmente | `np.where(cond, sim, nao)` |
| Multiplicação de matrizes | `A @ B` |
| Criar números aleatórios reproduzíveis | `rng = np.random.default_rng(42)` |
| Índice do maior valor (classe prevista) | `np.argmax(arr, axis=1)` |


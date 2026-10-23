#MEI-DAA #python #ML 

Se o NumPy é o motor de cálculo matemático, o **Pandas** é a bancada de trabalho onde passas 70% a 80% do teu tempo num projeto real de Ciência de Dados. 

Em Machine Learning, os dados raramente chegam como matrizes limpas de números. Chegam como tabelas confusas de clientes, vendas ou sensores — misturando textos, datas, números e muitos valores em falta. O Pandas serve exatamente para carregar, explorar, limpar, transformar e estruturar esses dados antes de os entregares aos algoritmos.

# Manual Prático de Pandas para Data Science & ML

```python
import pandas as pd
import numpy as np
```

## Módulo 1: Os Dois Blocos de Construção (`Series` e `DataFrame`)

Pensa no Pandas como uma folha de cálculo do Excel turbinada na memória RAM, dividida em dois conceitos fundamentais:

```
  Series (1D)                 DataFrame (2D)
   Índice  Valor           Índice   Nome      Idade   Salário
     0  ->  25               0   -> Ana        25      2400.0
     1  ->  30               1   -> Bruno      30      3100.0
     2  ->  22               2   -> Carla      22      1800.0
```

1. **`Series`**: É uma coluna individual (um array 1D do NumPy com rótulos/índices associados).
2. **`DataFrame`**: É a tabela completa (um conjunto de `Series` partilhando o mesmo índice de linhas).

```python
# Criar um DataFrame a partir de um dicionário (o método mais intuitivo)
dados = {
    'Nome': ['Ana', 'Bruno', 'Carla', 'Diogo'],
    'Idade': [28, 34, 22, 45],
    'Salario': [2500.0, 3800.0, 1900.0, 5200.0],
    'Comprou': [1, 0, 1, 0]  # O nosso futuro target (y)
}

df = pd.DataFrame(dados)
print(df)
```

## Módulo 2: O Raio-X Inicial do Dataset (EDA Rápida)

Quando recebes um dataset novo, **nunca** deves começar a treinar modelos sem correr este quarteto de comandos:

```python
# 1. Carregar dados de um ficheiro (ex: CSV)
# df = pd.read_csv('dados.csv')

# 2. Espreitar as primeiras e últimas linhas
print(df.head(2))   # Primeiras 2 amostras
print(df.tail(2))   # Últimas 2 amostras

# 3. Dimensões
print(df.shape)     # (4, 4) -> 4 linhas, 4 colunas

# 4. Tipos de dados e consumo de memória (CRUCIAL)
df.info()           # Mostra se há nulos e qual o tipo (int64, object/string, float64)

# 5. Resumo estatístico descritivo
print(df.describe()) # Média, desvio padrão, min, quartis, max (apenas colunas numéricas)
```

## Módulo 3: Seleção e Indexação — `loc` vs `iloc` de Vez!

Aqui nasce a maior parte da confusão dos iniciantes. A regra de ouro é:

* **`.iloc` (Integer-location):** Funciona estritamente por **posições numéricas** (exatamente como no NumPy: `[linha_idx, coluna_idx]`).
* **`.loc` (Label-location):** Funciona pelos **nomes/rótulos** das colunas e das linhas.

```python
# 1. Selecionar colunas diretamente:
idade_series = df['Idade']              # Retorna uma Series (1D)
sub_df = df[['Nome', 'Salario']]        # Retorna um DataFrame (passando lista de colunas)

# 2. Usando o .iloc (Pensa: "i" de inteiro)
# Ex: Quero as primeiras 2 linhas e as colunas de índice 1 e 2
print(df.iloc[0:2, 1:3])

# 3. Usando o .loc (Pensa: nomes das variáveis)
# Ex: Quero todas as linhas das colunas 'Nome' e 'Comprou'
print(df.loc[:, ['Nome', 'Comprou']])

# 💡 Como separar Features (X) e Target (y) em Pandas para ML:
X = df.drop(columns=['Comprou'])  # Tudo exceto a classe alvo
y = df['Comprou']                 # Apenas a coluna alvo
```

## Módulo 4: Filtragem Condicional e *Queries*

Como extrair fatias de interesse dos dados (ex: encontrar clientes de alto risco)?

```python
# 1. Filtro simples com máscara booleana
filtro_salario = df['Salario'] > 2000
print(df[filtro_salario])

# 2. Múltiplas condições (USA SEMPRE parênteses e operadores bitwise: & para E, | para OU)
clientes_alvo = df[(df['Idade'] > 25) & (df['Comprou'] == 1)]
print(clientes_alvo)

# 3. Filtrar com listas usando .isin()
nomes_procurados = ['Ana', 'Diogo']
print(df[df['Nome'].isin(nomes_procurados)])

# 4. Método .query() (Sintaxe muito legível e adorada em pipelines)
print(df.query("Idade < 30 and Salario > 2000"))
```


## Módulo 5: Limpeza de Dados e Tratamento de Nulos (`NaN`)

Modelos de Machine Learning do Scikit-Learn geralmente rebentam se receberem `NaN`. Tens de tratar deles:

```python
# Criar um DataFrame com falhas propositadas
df_falhas = pd.DataFrame({
    'Idade': [25.0, np.nan, 30.0, 45.0],
    'Score': [80.0, 95.0, np.nan, 70.0],
    'Cidade': ['Lisboa', 'Porto', np.nan, 'Lisboa']
})

# 1. Diagnóstico de Nulos
print(df_falhas.isna().sum())           # Quantidade de nulos por coluna
print(df_falhas.isna().mean() * 100)    # Percentagem de nulos por coluna

# 2. Eliminar linhas ou colunas com nulos (dropna)
df_limpo = df_falhas.dropna(axis=0)              # Apaga qualquer linha com pelo menos um NaN
df_limpo = df_falhas.dropna(subset=['Idade'])    # Apaga apenas se o NaN estiver na coluna 'Idade'

# 3. Imputação (Substituir nulos por estatísticas)
# Preencher idade nula com a mediana da idade:
mediana_idade = df_falhas['Idade'].median()
df_falhas['Idade'] = df_falhas['Idade'].fillna(mediana_idade)

# 4. Remover duplicados
df_falhas = df_falhas.drop_duplicates()
```


## Módulo 6: Engenharia de Atributos (*Feature Engineering*)

Criar novas variáveis a partir das existentes é o que separa um modelo medíocre de um modelo com alta precisão preditiva.

```python
df_vendas = pd.DataFrame({
    'Funcionario': ['Rita', 'Tomas', 'Leonor'],
    'Vendas_Q1': [10000, 15000, 8000],
    'Vendas_Q2': [12000, 14000, 9500]
})

# 1. Criação direta (Vetorizada e ultra-rápida)
df_vendas['Total_Vendas'] = df_vendas['Vendas_Q1'] + df_vendas['Vendas_Q2']
df_vendas['Crescimento'] = (df_vendas['Vendas_Q2'] - df_vendas['Vendas_Q1']) / df_vendas['Vendas_Q1']

# 2. Modificações em Texto com o acessor .str
df_vendas['Funcionario_Upper'] = df_vendas['Funcionario'].str.upper()

# 3. O método .apply() (Para regras condicionais personalizadas)
def classificar_desempenho(total):
    if total > 25000:
        return 'Excelente'
    return 'Regular'

df_vendas['Categoria'] = df_vendas['Total_Vendas'].apply(classificar_desempenho)
print(df_vendas)
```


## Módulo 7: Tratamento de Variáveis Categóricas

Computadores só entendem números. Se tens colunas como "Sexo", "País" ou "Nível", tens de as converter antes do treino.

### A. Avaliar Frequência:
```python
df_clientes = pd.DataFrame({
    'Escolaridade': ['Licenciatura', 'Mestrado', 'Licenciatura', 'Doutoramento', 'Secundário'],
    'Sexo': ['F', 'M', 'F', 'F', 'M']
})

# Ver proporções de cada categoria
print(df_clientes['Escolaridade'].value_counts(normalize=True))
```

### B. One-Hot Encoding com `pd.get_dummies`
Transforma texto nominal em colunas binárias ($0$ ou $1$):

```python
# drop_first=True evita o problema de multicolinearidade (Dummy Variable Trap)!
df_dummies = pd.get_dummies(df_clientes, columns=['Sexo'], drop_first=True, dtype=int)
print(df_dummies)
```

### C. Mapeamento Ordinal (Quando existe hierarquia)
```python
graus = {'Secundário': 1, 'Licenciatura': 2, 'Mestrado': 3, 'Doutoramento': 4}
df_clientes['Nivel_Num'] = df_clientes['Escolaridade'].map(graus)
print(df_clientes[['Escolaridade', 'Nivel_Num']])
```


## Módulo 8: Agrupamentos e Agregações (`groupby`)

O `groupby` segue o paradigma fundamental da Ciência de Dados: **Split $\rightarrow$ Apply $\rightarrow$ Combine**.

1. **Split:** Divide a tabela em grupos menores segundo uma categoria.
2. **Apply:** Calcula uma métrica (média, soma, desvio padrão) para cada grupo.
3. **Combine:** Junta tudo de volta num novo sumário.

```python
transacoes = pd.DataFrame({
    'Loja': ['Lisboa', 'Porto', 'Lisboa', 'Porto', 'Coimbra', 'Lisboa'],
    'Setor': ['Tecnologia', 'Moda', 'Moda', 'Tecnologia', 'Moda', 'Tecnologia'],
    'Receita': [500, 300, 150, 700, 200, 450]
})

# 1. Agregação Simples: Média de receita por loja
media_por_loja = transacoes.groupby('Loja')['Receita'].mean()
print(media_por_loja)

# 2. Agregação Múltipla com .agg() (Poderoso!)
resumo = transacoes.groupby('Loja').agg(
    Receita_Total=('Receita', 'sum'),
    Receita_Media=('Receita', 'mean'),
    Transacoes=('Receita', 'count')
)
print(resumo)
```


## Módulo 9: Combinar Datasets (`merge` e `concat`)

Raramente todos os teus dados vêm num único ficheiro.

### A. `pd.concat`: Empilhar dados
```python
df_2023 = pd.DataFrame({'Ano': [2023], 'Valor': [100]})
df_2024 = pd.DataFrame({'Ano': [2024], 'Valor': [120]})

# Empilhar linhas verticalmente (como rbind do R)
df_historico = pd.concat([df_2023, df_2024], ignore_index=True)
```

### B. `pd.merge`: Cruzar tabelas (O equivalente ao JOIN do SQL)
```python
usuarios = pd.DataFrame({
    'user_id': [1, 2, 3],
    'Nome': ['Pedro', 'Marta', 'João']
})

compras = pd.DataFrame({
    'user_id': [1, 1, 2],
    'Valor': [50.0, 80.0, 120.0]
})

# Left Join: Mantém todos os utilizadores, traz as compras se existirem
df_final = pd.merge(usuarios, compras, on='user_id', how='left')
print(df_final)
```


## Módulo 10: Ponte para o Machine Learning (`to_numpy` e Cuidados)

Chegaste ao fim da limpeza e exploração. O teu modelo do Scikit-Learn está à espera. Como fazes a transição final?

```python
# 1. Preparar a matriz de features e o vetor alvo
X = df[['Idade', 'Salario']]
y = df['Comprou']

# 2. Extrair arrays do NumPy de forma moderna
# 💡 NOTA: Usa sempre .to_numpy() em vez do antigo .values!
X_numpy = X.to_numpy()
y_numpy = y.to_numpy()

print(type(X_numpy))  # <class 'numpy.ndarray'>
print(X_numpy.shape)  # Pronto para train_test_split do Sklearn!
```

### ⚠️ O Alerta Clássico: `SettingWithCopyWarning`
Quantas vezes já viste um aviso vermelho a dizer `A value is trying to be set on a copy of a slice from a DataFrame`?
* **Porque acontece?** Fizeste uma fatia de um DataFrame e tentaste atribuir valores sem especificar se é cópia ou visualização.
* **A solução:** Se estás a criar um sub-conjunto de dados para alterar, força explicitamente o `.copy()`:
```python
# ❌ Evita:
# sub_df = df[df['Idade'] > 25]
# sub_df['Nova_Coluna'] = 10  # Gera aviso!

# ✅ Faz sempre:
sub_df = df[df['Idade'] > 25].copy()
sub_df['Nova_Coluna'] = 10     # Limpo e seguro!
```


## 📋 Folha de Consulta Rápida de Pandas (Cheat Sheet)

| Tarefa | Comando Pandas |
| :--- | :--- |
| **Primeiras Linhas** | `df.head(n)` |
| **Verificar Nulos** | `df.isna().sum()` |
| **Estatística Global** | `df.describe()` |
| **Seleção por Nome** | `df.loc[linhas, colunas]` |
| **Seleção por Posição** | `df.iloc[0:10, 0:3]` |
| **Filtrar Dados** | `df[(df['col1'] > 10) & (df['col2'] == 'A')]` |
| **Preencher Nulos** | `df['col'].fillna(df['col'].median())` |
| **Criar Dummies (One-Hot)** | `pd.get_dummies(df, drop_first=True)` |
| **Agrupamentos** | `df.groupby('chave')['valor'].mean()` |
| **Cruzamento (Join)** | `pd.merge(df1, df2, on='id', how='inner')` |
| **Exportar para NumPy** | `df.to_numpy()` |


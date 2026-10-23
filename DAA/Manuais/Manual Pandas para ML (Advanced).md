#MEI-DAA #python #ML 

## Módulo 11: Séries Temporais e Datas (`pd.to_datetime` e o acessor `.dt`)

Modelos de Machine Learning não compreendem datas no formato texto (ex: `"2024-03-15"`). O truque é converter a coluna para o tipo `datetime` e decompor essa data em várias *features* numéricas.

```python
df_temporal = pd.DataFrame({
    'Data_Venda': ['2024-01-15 09:30:00', '2024-06-20 18:45:00', '2024-12-25 12:00:00'],
    'Faturacao': [120.0, 340.0, 510.0]
})

# 1. Converter string para Datetime
df_temporal['Data_Venda'] = pd.to_datetime(df_temporal['Data_Venda'])

# 2. O acessor .dt: Extrair features temporais ricas para o modelo
df_temporal['Ano'] = df_temporal['Data_Venda'].dt.year
df_temporal['Mes'] = df_temporal['Data_Venda'].dt.month
df_temporal['Dia'] = df_temporal['Data_Venda'].dt.day
df_temporal['Dia_Semana'] = df_temporal['Data_Venda'].dt.dayofweek  # 0=Segunda, 6=Domingo
df_temporal['Hora'] = df_temporal['Data_Venda'].dt.hour

# 3. Criar uma flag binária de "Fim de Semana" (Feature Engineering!)
df_temporal['E_Fim_Semana'] = df_temporal['Dia_Semana'].isin([5, 6]).astype(int)

print(df_temporal[['Data_Venda', 'Mes', 'Dia_Semana', 'E_Fim_Semana']])
```

## Módulo 12: Ordenação e os "Top-K" (`sort_values`, `nlargest`, `nsmallest`)

Identificar rapidamente os maiores clientes, as transações mais suspeitas ou as piores perdas.

```python
df_clientes = pd.DataFrame({
    'Cliente': ['Marta', 'Tiago', 'Inês', 'Rui', 'Beatriz'],
    'Score': [750, 420, 890, 610, 890],
    'Idade': [30, 45, 29, 52, 40]
})

# 1. Ordenação Simples e Múltipla
# Ordena por Score descendente; em caso de empate, pela Idade ascendente
df_ordenado = df_clientes.sort_values(by=['Score', 'Idade'], ascending=[False, True])
print(df_ordenado)

# 2. Atalhos para os Top-K (Muito mais rápidos que ordenar o DataFrame inteiro!)
# Os 3 clientes com maior Score:
top_3_score = df_clientes.nlargest(3, 'Score')

# Os 2 clientes com menor Idade:
mais_jovens = df_clientes.nsmallest(2, 'Idade')
```

## Módulo 13: Discretização de Variáveis Contínuas (`pd.cut` e `pd.qcut`)

Muitas vezes, algoritmos funcionam melhor se agrupares idades ou salários em "baldes" (*bins* ou categorias).

* **`pd.cut`:** Divide os dados com base em **intervalos definidos por ti** (ou de largura fixa).
* **`pd.qcut` (Quantile Cut):** Divide os dados em baldes com o **mesmo número de amostras** em cada um (quartis, decis, etc.).

```python
idades = pd.Series([12, 17, 24, 38, 45, 62, 70, 85])

# A. pd.cut: Faixas etárias pré-definidas
faixas = [0, 18, 65, 100]
rotulos = ['Jovem', 'Adulto', 'Idoso']

categoria_idade = pd.cut(idades, bins=faixas, labels=rotulos, right=False)
print(categoria_idade)

# B. pd.qcut: 4 grupos equilibrados (Quartis: 25% dos dados em cada)
rendas = pd.Series([1000, 1200, 1500, 2000, 3000, 5000, 8000, 15000])
quartis_renda = pd.qcut(rendas, q=4, labels=['Q1_Baixa', 'Q2_Media_Baixa', 'Q3_Media_Alta', 'Q4_Alta'])
print(quartis_renda)
```

## Módulo 14: Matriz de Confusão Manual e Tabelas Cruzadas (`pd.crosstab`)

O `pd.crosstab` é uma das ferramentas mais subestimadas do Pandas. Ele cruza duas variáveis categóricas e conta as suas frequências.

### 💡 Caso de ML: Matriz de Confusão sem importar o Scikit-Learn!
```python
y_real = pd.Series([1, 0, 1, 1, 0, 0, 1, 0], name='Real')
y_pred = pd.Series([1, 0, 0, 1, 0, 1, 1, 0], name='Previsao')

# Tabela cruzada direta com percentagens
matriz = pd.crosstab(y_real, y_pred, margins=True, normalize='index')
print("Matriz de Confusão Normalizada:\n", matriz)
```

## Módulo 15: Reestruturação de Formato: `melt` vs `pivot` (Tidy Data)

Muitos gráficos no Seaborn exigem que os dados estejam no formato **Longo** (*Long/Tidy format*), mas os dados brutos chegam frequentemente no formato **Largo** (*Wide format*).

```
Formato Largo (Wide)             Formato Longo (Melted)
 Pais   2021  2022                Pais   Ano   PIB
  PT    100   105         ->       PT    2021  100
  ES    500   520                  PT    2022  105
                                   ES    2021  500
                                   ES    2022  520
```

```python
# 1. Transformar de Largo para Longo com pd.melt
df_wide = pd.DataFrame({
    'Pais': ['Portugal', 'Espanha'],
    'Ano_2022': [200, 1200],
    'Ano_2023': [210, 1250]
})

df_long = pd.melt(
    df_wide, 
    id_vars=['Pais'], 
    value_vars=['Ano_2022', 'Ano_2023'],
    var_name='Ano', 
    value_name='PIB'
)
print(df_long)

# 2. Operação inversa: Voltar ao formato Largo com .pivot
df_reconstruido = df_long.pivot(index='Pais', columns='Ano', values='PIB')
```

## Módulo 16: Gestão e Higiene de Índices (`reset_index` e `set_index`)

Quando filtras linhas ou fazes um `groupby`, o índice do DataFrame fica cheio de buracos ou adota um formato multinível (*MultiIndex*). Se não o limpares, terás problemas ao tentar juntar tabelas ou treinar modelos.

```python
df_teste = pd.DataFrame({'Categoria': ['A', 'B', 'C', 'D'], 'Valor': [10, 20, 30, 40]})

# Filtrar removeu linhas (os índices agora são 1 e 3)
df_filtrado = df_teste[df_teste['Valor'] > 15]

# ⚠️ O índice antigo fica [1, 3]!
# ✅ Limpeza essencial: reset_index
# drop=True evita que o índice antigo vire uma coluna extra
df_limpo = df_filtrado.reset_index(drop=True)
print(df_limpo)

# Definir uma coluna identificadora única como índice (ex: ID da transação)
df_com_id = df_limpo.set_index('Categoria')
```

## Módulo 17: Amostragem e Limpeza Fina (`sample`, `replace` e `clip`)

Operações cruciais de pré-processamento que poupam horas de trabalho:

### A. `df.sample`: Amostragem Aleatória (Bootstrap / Sub-amostragem)
```python
# Tirar uma amostra aleatória de 50% dos dados para testes rápidos
df_amostra = df_teste.sample(frac=0.5, random_state=42)

# Tirar exatamente 2 amostras com reposição
df_bootstrap = df_teste.sample(n=2, replace=True, random_state=42)
```

### B. `replace`: Tratar caracteres estranhos que representam Nulos
Muitas bases de dados usam `"?"`, `"-"` ou `"N/D"` para indicar valores em falta, fazendo com que o Pandas interprete a coluna como texto (*object*).

```python
df_sujo = pd.DataFrame({'Preco': ['10.5', '20.0', '?', '15.2', '-']})

# Substituir marcadores por np.nan real e converter para numérico
df_sujo['Preco'] = df_sujo['Preco'].replace(['?', '-'], np.nan).astype(float)
print(df_sujo['Preco'].dtype)  # float64 (pronto para contas!)
```

### C. `clip`: Limitar *Outliers* extremos
Em vez de apagares amostras com valores extremos, podes "aparar" os extremos:
```python
idades_com_erros = pd.Series([18, 25, 40, 190, 32, -5])

# Ninguém pode ter menos de 0 nem mais de 100 anos:
idades_validadas = idades_com_erros.clip(lower=0, upper=100)
print(idades_validadas)  # [18, 25, 40, 100, 32, 0]
```

## Módulo 18: Correlação Linear (`df.corr()` — O Melhor Amigo do Seaborn)

Antes de selecionar variáveis para um modelo supervisionado, deves verificar:
1. Quais as variáveis com **maior correlação** com a variável alvo ($y$).
2. Se existem variáveis independentes redundantes (multicolinearidade).

```python
df_ml = pd.DataFrame({
    'Area_m2': [50, 70, 85, 120, 150],
    'Quartos': [1, 2, 2, 3, 4],
    'Distancia_Centro_Km': [15, 10, 8, 3, 2],
    'Preco_Milhares': [150, 210, 260, 390, 480]
})

# Calcula a matriz de correlação de Pearson (-1 a +1)
# 💡 Usa numeric_only=True para garantir compatibilidade no Pandas moderno!
matriz_corr = df_ml.corr(numeric_only=True)
print(matriz_corr)

# Ver apenas a correlação das features com o Preço, ordenada:
correlacoes_com_target = matriz_corr['Preco_Milhares'].sort_values(ascending=False)
print("\nRelação com o Preço:\n", correlacoes_com_target)
```

## 📌 Tabela de Consulta Expandida (Cheat Sheet Completa)

| Desafio | A Solução em Pandas |
| :--- | :--- |
| **Extrair dia/mês/ano de texto** | `pd.to_datetime(col).dt.month` |
| **Pegar os 5 maiores valores** | `df.nlargest(5, 'coluna')` |
| **Dividir número contínuo em faixas** | `pd.cut(col, bins=[...], labels=[...])` |
| **Dividir número em 4 quantis iguais** | `pd.qcut(col, q=4)` |
| **Matriz de Confusão / Frequência cruzada** | `pd.crosstab(y_true, y_pred)` |
| **Passar colunas para formato longo** | `pd.melt(df, id_vars=[...])` |
| **Redefinir índice após filtros** | `df.reset_index(drop=True)` |
| **Trocar strings anómalas por NaN** | `df['col'].replace('?', np.nan)` |
| **Limitar valores extremos (Outliers)** | `df['col'].clip(lower, upper)` |
| **Calcular correlações entre colunas** | `df.corr(numeric_only=True)` |
| **Sortear amostra aleatória** | `df.sample(frac=0.2, random_state=42)` |



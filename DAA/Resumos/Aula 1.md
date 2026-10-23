## 1. Tipos de Dados (*Data Types*)

Saber o tipo de dado determina o pré-processamento e o algoritmo a aplicar:

| Tipo                     | Descrição                                                                                                           | Exemplos                                                                                   |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- |
| **Numérico Discreto**    | Valores inteiros finitos/contáveis (número de eventos).                                                             | N.º de compras, n.º de caras ao lançar moeda, idade em anos inteiros.                      |
| **Numérico Contínuo**    | Valores reais com infinitas possibilidades de medição dentro de um intervalo.                                       | Tempo de resposta, pluviosidade, preço, gasolina no tanque.                                |
| **Categórico (Nominal)** | Qualitativo sem relação matemática de ordem. Rótulos numéricos (se atribuídos) servem apenas de código.             | Género, país/nacionalidade, tipo de produto.                                               |
| **Ordinal**              | Categórico com relação de ordem/hierarquia natural, embora as distâncias numéricas não sejam estritamente métricas. | Avaliação de filmes (1 a 5 estrelas), estado de saúde (*Poor, Moderate, Good, Excellent*). |

## 2. Medidas de Tendência Central

* **Média (*Mean*)**: Soma de todas as observações a dividir pelo número total ($N$).  
  $$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$$  
  *Ponto fraco:* É muito sensível a *outliers* (valores extremos).
* **Mediana (*Median*)**: O valor central após ordenação dos dados (ou média dos dois centrais se $N$ for par).  
  *Ponto forte:* **Robusta a *outliers*** (ex.: salários, onde multimilionários distorcem a média, mas não a mediana).
* **Moda (*Mode*)**: Valor que ocorre com maior frequência. Aplica-se sobretudo a dados discretos ou categóricos.

## 3. Dispersão: Variância e Desvio Padrão

Medem o grau de afastamento (*spread*) dos dados em torno da média:

1. **Variância ($\sigma^2$)**: Média dos desvios quadráticos face à média:
   $$\sigma^2 = \frac{\sum (x_i - \mu)^2}{N}$$
2. **Desvio Padrão ($\sigma$)**: Raiz quadrada da variância ($\sigma = \sqrt{\sigma^2}$).  
   *Expressa a dispersão na mesma unidade de medida dos dados originais.* É a base para identificar anomalias (*outliers* com base na distância de $\pm 1\sigma, \pm 2\sigma, \pm 3\sigma$).

## 4. Distribuições e Funções de Probabilidade

* **PMF (*Probability Mass Function*)**: Utilizada para **variáveis discretas**. Devolve diretamente a probabilidade de um valor exato ocorrer: $P(X = k)$.
* **PDF (*Probability Density Function*)**: Utilizada para **variáveis contínuas** (ex.: Distribuição Normal / Gaussiana). A probabilidade de um valor exato é zero; a probabilidade é calculada como a **área sob a curva** dentro de um determinado intervalo.

## 5. Percentis e Intervalo Interquartil (IQR)

* **Percentil $X$**: Valor abaixo do qual se encontra $X\%$ das amostras (ex.: o percentil 99 de rendimentos indica o valor que 99% da população ganha abaixo).
* **Quartis**:
  * $Q_1$ = Percentil 25
  * $Q_2$ = Mediana (Percentil 50)
  * $Q_3$ = Percentil 75
* **IQR (*Inter-Quartile Range*)**: $IQR = Q_3 - Q_1$.  
  Representa os 50% centrais dos dados (zona de maior densidade), servindo de base clássica (no *Boxplot*) para detetar *outliers*: valores fora do intervalo $[Q_1 - 1.5 \times IQR,\; Q_3 + 1.5 \times IQR]$.

## 6. Relações entre Variáveis: Covariância vs. Correlação

* **Covariância**: Mede se duas variáveis variam na mesma direção.
  $$\text{Cov}(X, Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{N}$$
  *Problema:* Depende da escala das variáveis, tornando difícil avaliar a intensidade real da relação.
* **Correlação de Pearson ($r$)**: Covariância normalizada pelos desvios padrão:
  $$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
  *Interpretação:*
  * $+1$: Correlação linear positiva perfeita.
  * $0$: Ausência de correlação linear.
  * $-1$: Correlação linear negativa/inversa perfeita.
* **Regra de ouro:** *«Correlação não implica Causalidade»* — relações lineares apenas sugerem hipóteses que devem ser validadas com testes controlados e aleatórios.

## 7. Ambientes Virtuais (Virtual Environments)

Evitam conflitos de versões entre dependências de projetos distintos na mesma máquina.

### Comandos Essenciais — Conda / Miniconda

| Ação | Comando |
| :--- | :--- |
| **Criar ambiente** | `conda create --name meu_env python=3.14 numpy` |
| **Ativar ambiente** | `conda activate meu_env` |
| **Instalar pacotes** | `conda install pandas scikit-learn` |
| **Listar ambientes** | `conda info --envs` |
| **Listar bibliotecas instaladas** | `conda list` |
| **Desativar ambiente** | `conda deactivate` |
| **Eliminar ambiente** | `conda env remove --name meu_env` |

### Comandos Essenciais — `venv` (Nativo do Python)

| Ação                    | macOS / Linux                   | Windows (PowerShell / CMD)                     |
| :---------------------- | :------------------------------ | :--------------------------------------------- |
| **Criar**               | `python -m venv meu_env`        | `python -m venv meu_env`                       |
| **Ativar**              | `source meu_env/bin/activate`   | `.\meu_env\Scripts\Activate.ps1` *(ou `.bat`)* |
| **Instalar pacotes**    | `pip install <pacote>`          | `pip install <pacote>`                         |
| **Exportar requisitos** | `pip freeze > requirements.txt` | `pip freeze > requirements.txt`                |
| **Desativar**           | `deactivate`                    | `deactivate`                                   |

## 8. O *Stack* Essencial de Machine Learning em Python

Conjunto de ferramentas para o ciclo de desenvolvimento em ML (Questões T2 e T4):

1. **NumPy**: Álgebra linear, matrizes multidimensionais (*ndarrays*) e operações vetorizadas ultra-rápidas. É a base matemática de quase todas as ferramentas de ML.
2. **Pandas**: Manipulação e análise estruturada de dados tabulares (*DataFrames*), limpeza de dados e agregação.
3. **Matplotlib**: Criação de gráficos e visualização básica/baixo nível (histogramas, dispersão, etc.).
4. **Seaborn**: Visualização estatística de alto nível sobreposta ao Matplotlib (mapas de calor de correlação, *pairplots*, distribuições).
5. **Scikit-Learn**: Biblioteca nuclear para Machine Learning clássico: algoritmos de classificação, regressão, *clustering*, redução de dimensionalidade e métricas de validação.
6. **JupyterLab / Jupyter Notebook**: Ambiente interativo para prototipagem rápida, exploração de dados (EDA) e documentação combinada com código.
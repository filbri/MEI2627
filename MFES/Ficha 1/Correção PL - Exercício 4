### 1. Codificação do Feature Model em Lógica Proposicional

Vamos associar a cada funcionalidade do diagrama uma variável proposicional booleana:
* $S$ : `survey` (Raiz)
* $L$ : `license`
* $AL$ : `advanced license`
* $BL$ : `basic license`
* $AB$ : `AB testing`
* $Stat$ : `statistics`
* $QA$ : `QA`
* $BQA$ : `basic QA`
* $MQA$ : `multimedia QA`

A semântica de um Feature Diagram traduz-se segundo quatro regras canónicas:

#### (1) A Raiz é Obrigatória
Num produto da linha, a raiz tem de ser sempre selecionada:
$$\phi_{\text{root}} \equiv S$$

#### (2) Relações Pai-Filho (Estrutura da Árvore)
* **Presença de um filho implica a presença do pai:** nenhuma subfuncionalidade existe sem o seu conceito-pai:
  $$\phi_{\text{hierarchy}} \equiv (L \implies S) \land (AB \implies S) \land (Stat \implies S) \land (QA \implies S)$$

* **Funcionalidades Obrigatórias (*Mandatory* - círculo fechado $\bullet$):**
  Se o pai for selecionado, o filho obrigatório **tem** de ser incluído.
  $L$ e $QA$ são obrigatórias relativamente a $S$:
  $$\phi_{\text{mand}} \equiv (S \implies L) \land (S \implies QA)$$
  *(Nota: As funcionalidades $AB$ e $Stat$ têm círculo aberto $\circ$, logo são opcionais: não impõem implicação a partir de $S$.)*

#### (3) Grupos de Funcionalidades (*Feature Groups*)
* **Grupo Alternativo / XOR sob `license` (arco simples):**
  A licença exige **exatamente uma** das opções ($AL$ ou $BL$):
  1. O filho implica o pai: $(AL \implies L) \land (BL \implies L)$
  2. Pelo menos uma: $L \implies (AL \lor BL)$
  3. No máximo uma: $\neg(AL \land BL) \equiv (\neg AL \lor \neg BL)$
  $$\phi_{\text{license}} \equiv (L \iff (AL \lor BL)) \land (\neg AL \lor \neg BL)$$

* **Grupo Inclusivo / OR sob `QA` (triângulo/arco preenchido $\blacktriangle$):**
  O enunciado confirma: *"pelo menos uma destas tem de ser incluída"*.
  1. O filho implica o pai: $(BQA \implies QA) \land (MQA \implies QA)$
  2. Pelo menos uma: $QA \implies (BQA \lor MQA)$
  $$\phi_{QA} \equiv QA \iff (BQA \lor MQA)$$

#### (4) Restrições Transversais (*Cross-Tree Constraints*)
As setas tracejadas estabelecem dependências diretas entre ramos:
* **`AB testing` requires `statistics`:**
  $$\phi_{\text{req}} \equiv AB \implies Stat \quad \equiv \quad \neg AB \lor Stat$$
* **`basic license` excludes `AB testing`:**
  $$\phi_{\text{ex1}} \equiv \neg (BL \land AB) \quad \equiv \quad \neg BL \lor \neg AB$$
* **`basic license` excludes `multimedia QA`:**
  $$\phi_{\text{ex2}} \equiv \neg (BL \land MQA) \quad \equiv \quad \neg BL \lor \neg MQA$$

---

#### Modelo Lógico Global ($\Phi$)
A especificação formal do Feature Model é a conjunção de todas estas regras:
$$\Phi = \phi_{\text{root}} \land \phi_{\text{hierarchy}} \land \phi_{\text{mand}} \land \phi_{\text{license}} \land \phi_{QA} \land \phi_{\text{req}} \land \phi_{\text{ex1}} \land \phi_{\text{ex2}}$$

---

### 2. Análise do Modelo com o SAT Solver

Vejamos como formular cada questão analítica em perguntas de satisfiabilidade que o SAT solver compreende.

---

#### (a) O feature model é vazio? (*Void Feature Model*)
* **Conceito:** Um modelo diz-se **vazio** se não admitir nenhuma configuração válida de produto (isto é, se a especificação for contraditória).
* **Uso do SAT Solver:** 
  Submete-se simplesmente a fórmula $\Phi$ ao solver.
  * Se devolver **`UNSAT`**, o modelo é **vazio** (erro de desenho de software).
  * Se devolver **`SAT`**, o modelo **não é vazio** (é consistente e válido).
* *No nosso caso:* É **SAT** (por exemplo, $\{S, L, BL, QA, BQA\}$ é um produto válido), logo o modelo não é vazio.

---

#### (b) Quantas variantes existem? (*Model Counting / #SAT*)
* **Conceito:** O número de variantes corresponde ao número total de produtos distintos (modelos booleanos) que satisfazem $\Phi$.
* **Uso do SAT Solver:**
  * **Via Solver Padrão (com *Blocking Clauses*):** 
    Inicia-se um contador em $0$. Em ciclo, enquanto o solver retornar **`SAT`**, recolhe-se o modelo $M$, incrementa-se o contador, e injeta-se na base de conhecimento uma cláusula de bloqueio que nega essa atribuição: $\Phi \leftarrow \Phi \land \neg M$. Quando o solver responder finalmente **`UNSAT`**, o contador terá o número exato de variantes.
  * **Via #SAT Solver:** Recorrendo a ferramentas especializadas em contagem exata de modelos (*model counters* como o SharpSAT ou a funcionalidade de contagem do Z3).
* *Curiosidade de cálculo:*
  * Se escolhermos `basic license`: $AB=0$, $MQA=0$, restando $BQA=1$. Como $Stat$ pode ser $0$ ou $1$, temos **2 variantes**.
  * Se escolhermos `advanced license`: podemos ter qualquer combinação válida de QA (3 opções: só BQA, só MQA, ou ambos) e de $(AB, Stat)$ sujeitas a $AB \implies Stat$ (3 opções: $(0,0), (0,1), (1,1)$). Como são independentes, $3 \times 3 =$ **9 variantes**.
  * **Total: 11 variantes possíveis.**

---

#### (c) Se o “AB testing” fosse obrigatório, quais seriam as funcionalidades nucleares? (*Core Features*)
* **Conceito:** Uma funcionalidade é dita **nuclear** (*core feature*) se estiver presente em **todas** as variantes válidas do produto.
* **Uso do SAT Solver:**
  1. Criamos a teoria aumentada: $\Phi' = \Phi \land AB$.
  2. Para cada funcionalidade $f$ do modelo, verificamos se é impossível criar um produto sem ela, recorrendo ao teorema da refutação:
     $$\text{Testar: } \Phi' \land \neg f$$
     * Se o solver responder **`UNSAT`**, significa que é impossível omitir $f$, logo **$f$ é uma funcionalidade nuclear**!
     * Se responder **`SAT`**, $f$ é opcional.
* *Análise formal:*
  * Sob $\Phi'$, como $AB = \top$, a regra de exclusão $\neg BL \lor \neg AB$ força $BL = \bot$.
  * Com $BL = \bot$, a regra da licença força $AL = \top$. Logo, **$AL$ torna-se nuclear**!
  * Como $AB = \top$, a regra $AB \implies Stat$ força $Stat = \top$. Logo, **$Stat$ torna-se nuclear**!
  * As funcionalidades nucleares seriam: **$\{survey, license, QA, AB\text{ testing}, advanced\text{ license}, statistics\}$**.

---

#### (d) Se o “QA basic” exigisse o “AB testing”, haveria funcionalidades mortas? (*Dead Features*)
* **Conceito:** Uma funcionalidade é **morta** (*dead feature*) se nunca puder aparecer em nenhum produto válido (está presente no diagrama, mas é impossível de instanciar sem violar regras).
* **Uso do SAT Solver:**
  1. Definimos a nova teoria: $\Phi'' = \Phi \land (BQA \implies AB)$.
  2. Para cada funcionalidade $f$, perguntamos ao solver se é possível construir um produto onde $f$ esteja ativa:
     $$\text{Testar: } \Phi'' \land f$$
     * Se o solver responder **`UNSAT`**, então **$f$ é uma funcionalidade morta**!
     * Se responder **`SAT`**, $f$ está viva (é utilizável).
* *Análise formal — Haverá mortas?*
  **Sim! A funcionalidade `basic license` ($BL$) fica morta!**
  Repara na cadeia de deduções lógicas que o solver faz automaticamente:
  1. Supõe que tentamos selecionar $BL = \top$.
  2. Pela regra de exclusão, $BL \implies \neg MQA$, logo $MQA = \bot$.
  3. Mas o grupo $QA$ obriga a $BQA \lor MQA$. Como $MQA = \bot$, somos forçados a ter $BQA = \top$.
  4. Pela nova regra, $BQA \implies AB$, logo temos de ter $AB = \top$.
  5. Contudo, a regra original diz que $BL$ e $AB$ excluem-se mutuamente! Temos uma contradição flagrante ($BL \land AB$).
  6. Conclusão: $\Phi'' \land BL$ é **`UNSAT`**. A funcionalidade **`basic license` é uma funcionalidade morta**!
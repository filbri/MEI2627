
# Exercício 1 (Colocação de paineis)
## 0. Domínio

* **Posições:** $p \in P = \{1, 2, 3, 4, 5, 6, 7\}$
* **Materiais:** $m \in M = \{1, 2, 3\}$, onde:
  * $1 \rightarrow \text{Madeira}$
  * $2 \rightarrow \text{Metal}$
  * $3 \rightarrow \text{Vidro}$
* **Variáveis proposicionais:** $x_{p,m}$ significa *"o painel na posição $p$ é feito do material $m$"*.


## 1. Restrições
### (a) Cada uma das 7 posições possíveis tem que ter um painel
**Intuição:** Em cada posição $p$, temos de garantir que é atribuído *pelo menos um* material.

$$\bigwedge_{p=1}^{7} (x_{p,1} \lor x_{p,2} \lor x_{p,3})$$

*Nota: Garante que nenhuma posição fica vazia.*

### (b) Os painéis não podem ser sobrepostos
**Intuição:** Nenhuma posição $p$ pode ter dois ou mais materiais em simultâneo (unicidade). Ou seja, em cada posição temos *no máximo um* material. Para cada par de materiais distintos, não podemos ter ambos ativos ao mesmo tempo:

$$\bigwedge_{p=1}^{7} \left( \neg(x_{p,1} \land x_{p,2}) \land \neg(x_{p,1} \land x_{p,3}) \land \neg(x_{p,2} \land x_{p,3}) \right)$$

Pela *Lei de De Morgan* [^1], podemos reescrever diretamente como cláusulas:

$$\bigwedge_{p=1}^{7} \left( (\neg x_{p,1} \lor \neg x_{p,2}) \land (\neg x_{p,1} \lor \neg x_{p,3}) \land (\neg x_{p,2} \lor \neg x_{p,3}) \right)$$

*Nota: A junção de (a) e (b) garante a propriedade "**exact-one-hot**"[^2], isto é, cada posição tem exatamente um material.)*

### (c) Os painéis das extremidades não podem ser do mesmo tipo de material
**Intuição:** As extremidades são as posições $1$ e $7$. Para cada material $m \in \{1,2,3\}$, não é permitido que esteja simultaneamente na posição $1$ e na posição $7$:

$$\bigwedge_{m=1}^{3} \neg (x_{1,m} \land x_{7,m}) \quad \equiv \quad \bigwedge_{m=1}^{3} (\neg x_{1,m} \lor \neg x_{7,m})$$

Expandindo:
$$(\neg x_{1,1} \lor \neg x_{7,1}) \land (\neg x_{1,2} \lor \neg x_{7,2}) \land (\neg x_{1,3} \lor \neg x_{7,3})$$

Alternativamente:
$$(x_{1,1} \implies \neg x_{7,1}) \land (x_{1,2} \implies \neg x_{7,2}) \land (x_{1,3} \implies \neg x_{7,3})$$

### (d) O painel do meio não pode ser de vidro
**Intuição:** A posição central é $p = 4$. O material vidro corresponde a $m = 3$. Simplesmente negamos essa atribuição:

$$\neg x_{4,3}$$

### (e) Todos os materiais têm que ser usados
**Intuição:** Cada material $m \in \{1, 2, 3\}$ tem de aparecer em *pelo menos uma* das 7 posições:

$$\bigwedge_{m=1}^{3} \left( \bigvee_{p=1}^{7} x_{p,m} \right)$$

Expandindo para cada material:
1. Madeira é usada: $(x_{1,1} \lor x_{2,1} \lor x_{3,1} \lor x_{4,1} \lor x_{5,1} \lor x_{6,1} \lor x_{7,1})$
2. Metal é usado: $(x_{1,2} \lor x_{2,2} \lor x_{3,2} \lor x_{4,2} \lor x_{5,2} \lor x_{6,2} \lor x_{7,2})$
3. Vidro é usado: $(x_{1,3} \lor x_{2,3} \lor x_{3,3} \lor x_{4,3} \lor x_{5,3} \lor x_{6,3} \lor x_{7,3})$

### (f) Não podem ficar dois painéis de vidro seguidos
**Intuição:** Para quaisquer duas posições adjacentes $p$ e $p+1$, não podemos ter simultaneamente vidro ($m=3$):

$$\bigwedge_{p=1}^{6} \neg (x_{p,3} \land x_{p+1,3}) \quad \equiv \quad \bigwedge_{p=1}^{6} (\neg x_{p,3} \lor \neg x_{p+1,3})$$
Alternativamente:
$$\bigwedge_{p=1}^{6} x_{p,3} \implies \neg x_{p+1,3}$$

### (g) Junto a um painel de metal terá de haver pelo menos um painel de madeira
**Intuição:** Aqui temos de ter especial atenção às **condições de fronteira** (as extremidades têm apenas um vizinho; as posições interiores têm dois vizinhos).
Se na posição $p$ tivermos metal ($x_{p,2}$), a vizinhança tem de conter madeira ($m=1$).

* **Extremidade esquerda ($p=1$):** O único vizinho é a posição 2:
  $$x_{1,2} \implies x_{2,1} \quad \equiv \quad \neg x_{1,2} \lor x_{2,1}$$

* **Extremidade direita ($p=7$):** O único vizinho é a posição 6:
  $$x_{7,2} \implies x_{6,1} \quad \equiv \quad \neg x_{7,2} \lor x_{6,1}$$

* **Posições interiores ($p \in \{2, 3, 4, 5, 6\}$):** Tem pelo menos um vizinho de madeira (à esquerda ou à direita):
  $$\bigwedge_{p=2}^{6} \left( x_{p,2} \implies (x_{p-1,1} \lor x_{p+1,1}) \right) \quad \equiv \quad \bigwedge_{p=2}^{6} (\neg x_{p,2} \lor x_{p-1,1} \lor x_{p+1,1})$$

### (h) Se o primeiro painel for de vidro e o segundo de madeira, o último terá que ser de metal
**Intuição:** Uma implicação clássica com premissa composta:
$$(x_{1,3} \land x_{2,1}) \implies x_{7,2}$$

Na forma normal conjuntiva (recorrendo à equivalência $A \land B \implies C \equiv \neg A \lor \neg B \lor C$):
$$\neg x_{1,3} \lor \neg x_{2,1} \lor x_{7,2}$$

## 2. Uso do SAT Solver e Exploração de Múltiplas Soluções

#### Como tirar partido de um SAT solver para encontrar uma distribuição?
Um *SAT solver* (como o Z3, MiniSat ou CaDiCaL) decide se uma fórmula em lógica proposicional tem solução (**SAT**) ou se é contraditória (**UNSAT**).

O fluxo de trabalho é:
1. **Compilação:** Pegamos na fórmula global $\Phi = \phi_{(a)} \land \dots \land \phi_{(h)}$ (que já deduzimos estar em CNF[^2]) e codificamo-la no formato padrão (por exemplo, DIMACS CNF ou numa API de alto nível como Python/Z3).
2. **Execução:** Submetemos $\Phi$ ao motor do SAT solver.
3. **Interpretação:** 
   * Se o resultado for **UNSAT**, a nossa especificação é inconsistente — as restrições são tão rígidas que nenhuma combinação de painéis é fisicamente possível.
   * Se for **SAT**, o solver devolve uma **valoração** (um *modelo* $M \models \Phi$). Basta inspecionar quais as variáveis $x_{p,m}$ com valor de verdade `true`. Como garantimos nas alíneas (a) e (b) que existe exatamente um material por posição, obtemos a decoração completa da parede imediatamente!

#### Há várias alternativas? Como explorá-las?
Sim! 
Um problema combinatório com 7 posições e 3 materiais tem um espaço de busca inicial de $3^7 = 2187$ configurações possíveis. As regras eliminam a maioria, mas é expectável que sobrem várias distribuições válidas.

Para encontrar **todas** as soluções (ou navegar entre elas), utilizamos uma técnica clássica em métodos formais chamada **Cláusulas de Bloqueio (*Blocking Clauses*)**:

1. Pedimos a primeira solução ao solver: encontramos um modelo $M_1$.
2. Para cada posição $p$, seja $m_p$ o material atribuído no modelo $M_1$. A solução $M_1$ é a conjunção:
   $$x_{1, m_1} \land x_{2, m_2} \land \dots \land x_{7, m_7}$$
3. Para impedir o solver de nos devolver a mesma resposta, criamos uma **cláusula de bloqueio** negando essa atribuição completa:
   $$\neg (x_{1, m_1} \land x_{2, m_2} \land \dots \land x_{7, m_7}) \quad \equiv \quad (\neg x_{1, m_1} \lor \neg x_{2, m_2} \lor \dots \lor \neg x_{7, m_7})$$
4. Adicionamos esta cláusula à base de conhecimento:
   $$\Phi \leftarrow \Phi \land (\neg x_{1, m_1} \lor \dots \lor \neg x_{7, m_7})$$
5. Executamos novamente o solver:
   * Se devolver **SAT**, temos uma nova distribuição $M_2 \neq M_1$. Repetimos o processo!
   * Quando finalmente devolver **UNSAT**, temos a **prova formal** de que esgotámos e enumerámos rigorosamente todas as soluções válidas.

## 3. Descrição Lógica das Afirmações

A primeira fala de uma **obrigação/necessidade** ("terá de haver"), enquanto a segunda fala de uma **possibilidade**.

#### (a) *"Nas duas primeiras posições terá de haver, pelo menos, um painel de vidro."*
* A propriedade em si (seja ela $\psi_a$) afirma que a posição 1 tem vidro ($m=3$) OU a posição 2 tem vidro ($m=3$):
  $$\psi_a \equiv x_{1,3} \lor x_{2,3}$$
* Como afirmação sobre o sistema, o que o enunciado questiona é se esta regra decorre inevitavelmente das regras originais $\Phi$, ou seja, se $\psi_a$ é uma **consequência lógica**:
  $$\Phi \models (x_{1,3} \lor x_{2,3})$$

#### (b) *"É possível colocar um painel de madeira na última posição."*
* A propriedade de a última posição ser de madeira ($m=1$) é simplesmente a variável:
  $$\psi_b \equiv x_{7,1}$$
* No entanto, a frase *"é possível"* não é uma implicação, mas sim uma afirmação meta-lógica de **consistência / satisfiabilidade**: afirma que colocar madeira na posição 7 não entra em conflito com o conjunto de regras $\Phi$. Ou seja, existe pelo menos um modelo de $\Phi$ onde $x_{7,1}$ é verdadeiro.


## 4. Como testar a veracidade das afirmações no SAT Solver?

Aqui reside uma das distinções mais belas da Verificação Formal: **Necessidade (*Validity*) vs. Possibilidade (*Satisfiability*)**.

Um SAT solver responde nativamente a *"Isto é possível?"* (SAT/UNSAT). Mas como respondemos a *"Isto acontece obrigatoriamente em todos os casos?"*?

#### Testar a afirmação 3.(a) — Consequência Lógica / Validade
Queremos provar que $\Phi \implies \psi_a$ é uma tautologia (é válida para todas as decorações possíveis).
Pelo **Princípio da Refutação** (ou redução ao absurdo):
$$\Phi \models \psi_a \iff \Phi \land \neg \psi_a \text{ é INSATISFIÁVEL (UNSAT)}$$

1. Construímos a negação de $\psi_a$:
   $$\neg \psi_a \equiv \neg (x_{1,3} \lor x_{2,3}) \equiv \neg x_{1,3} \land \neg x_{2,3}$$
   *(Estamos a dizer ao solver: "Procura uma decoração válida onde NEM a posição 1 NEM a posição 2 tenham vidro".)*
2. Executamos o solver com:
   $$\Phi \land \neg x_{1,3} \land \neg x_{2,3}$$
3. Conclusão:
   * **Se o solver responder UNSAT:** A afirmação (a) é **VERDADEIRA**. Foi provado formalmente que é impossível violar a propriedade mantendo as regras do problema.
   * **Se o solver responder SAT:** A afirmação (a) é **FALSA**. O modelo devolvido pelo solver funciona como um **contra-exemplo** perfeito para desmentir a afirmação!

#### Testar a afirmação 3.(b) — Consistência / Possibilidade
Aqui queremos apenas saber se o cenário é viável em pelo menos uma distribuição válida. Não queremos negar a propriedade, queremos afirmá-la!

1. Adicionamos a hipótese $x_{7,1}$ diretamente à nossa especificação:
   $$\Phi \land x_{7,1}$$
2. Executamos o solver.
3. Conclusão:
   * **Se o solver responder SAT:** A afirmação (b) é **VERDADEIRA**. O modelo fornecido pelo solver é a **testemunha (*witness*)**, exibindo a parede completa que prova a possibilidade.
   * **Se o solver responder UNSAT:** A afirmação (b) é **FALSA**. As restrições do problema proíbem deterministicamente que a posição 7 seja de madeira.




# Referências

[^1]: Lei de Morgan: A negação de "P e Q" equivale a "não P ou não Q". Fórmula: $\neg (p \land q) \equiv \neg p \lor \neg q$
[^2]: Exact-One-Shot: Em lógica proposicional, exact-one-shot significa que exatamente uma das proposições é verdadeira (e todas as outras são falsas).
[^3]: CNF - Conjunctive Normal Form - Conjunctive Normal Form (CNF) é uma fórmula lógica que consiste numa conjunção (AND) de cláusulas, em que cada cláusula é uma disjunção (OR) de proposições ou das suas negações.

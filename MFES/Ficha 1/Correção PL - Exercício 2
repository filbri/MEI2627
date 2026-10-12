
# Exercício 2 (Distribuição de gabinetes)
## 0. Domínio

Primeiro, definimos os nossos conjuntos finitos:
* **Conjunto de Pessoas:** $\mathcal{P} = \{A, N, P, M\}$ 
  *(onde $A = \text{Ana}$, $N = \text{Nuno}$, $P = \text{Pedro}$, $M = \text{Maria}$)*
* **Conjunto de Gabinetes:** $\mathcal{G} = \{1, 2, 3\}$

Definimos agora as **variáveis proposicionais**:
> Seja $x_{i, g}$ a variável proposicional que é **verdadeira** se e só se a pessoa $i$ ocupa o gabinete $g$, para todo o $i \in \mathcal{P}$ e $g \in \mathcal{G}$.

Temos um total de $4 \times 3 = 12$ variáveis proposicionais:
$$\{x_{A,1}, x_{A,2}, x_{A,3}, x_{N,1}, x_{N,2}, x_{N,3}, x_{P,1}, x_{P,2}, x_{P,3}, x_{M,1}, x_{M,2}, x_{M,3}\}$$

## 1. Restrições
### (a) Cada pessoa ocupa um único gabinete
**Intuição:** Cada pessoa $i \in \mathcal{P}$ tem de estar em **pelo menos um** gabinete (existência) e em **no máximo um** gabinete (unicidade). Ou seja, exatamente um (*exact-one*).

1. **Pelo menos um gabinete por pessoa:**
   $$\bigwedge_{i \in \mathcal{P}} (x_{i,1} \lor x_{i,2} \lor x_{i,3})$$

2. **No máximo um gabinete por pessoa** (nenhuma pessoa pode estar em dois gabinetes distintos em simultâneo):   $$\bigwedge_{i \in \mathcal{P}} \left( (\neg x_{i,1} \lor \neg x_{i,2}) \land (\neg x_{i,1} \lor \neg x_{i,3}) \land (\neg x_{i,2} \lor \neg x_{i,3}) \right)$$
Alternativa:
$$\bigwedge_{i\in \mathcal{P}} (x_{i,1} \land \neg x_{i,2} \land \neg x_{i,3}) \lor (\neg x_{i,1} \land x_{i,2} \land \neg x_{i,3}) \lor (\neg x_{i,1} \land \neg x_{i,2} \land x_{i,3})$$

### (b) O Nuno e o Pedro não podem partilhar gabinete.
**Intuição:** Para qualquer gabinete $g \in \{1, 2, 3\}$, o Nuno e o Pedro não podem estar lá ao mesmo tempo:

$$\bigwedge_{g=1}^{3} \neg (x_{N,g} \land x_{P,g}) \quad \equiv \quad \bigwedge_{g=1}^{3} (\neg x_{N,g} \lor \neg x_{P,g})$$

Expandindo:
$$(\neg x_{N,1} \lor \neg x_{P,1}) \land (\neg x_{N,2} \lor \neg x_{P,2}) \land (\neg x_{N,3} \lor \neg x_{P,3})$$

Alternativamente:
$$\bigwedge_{g=1}^{3} \left (x_{2,g} \implies \neg x_{3,g} \right)$$

### (c) Se a Ana ficar sozinha num gabinete, então o Pedro também terá que ficar sozinho num gabinete.
**Intuição:** Esta restrição envolve o conceito de "estar sozinho".
O que significa uma pessoa $i$ estar sozinha num determinado gabinete $g$? Significa que a pessoa $i$ está em $g$ e **nenhuma outra pessoa** está em $g$:
$$\text{Sozinho}(i, g) \equiv x_{i,g} \land \bigwedge_{j \in \mathcal{P} \setminus \{i\}} \neg x_{j,g}$$

Assim, uma pessoa $i$ estar sozinha em **algum** gabinete é a disjunção sobre todos os gabinetes possíveis:
$$\text{Sozinha}(A) \equiv \bigvee_{g=1}^{3} \left( x_{A,g} \land \bigwedge_{j \in \{N, P, M\}} \neg x_{j,g} \right)$$
$$\text{Sozinho}(P) \equiv \bigvee_{g=1}^{3} \left( x_{P,g} \land \bigwedge_{j \in \{A, N, M\}} \neg x_{j,g} \right)$$

A regra é uma implicação lógica direta:
$$\text{Sozinha}(A) \implies \text{Sozinho}(P)$$
$$\left( \bigvee_{g=1}^{3} (x_{g,1} \land \neg x_{g,2} \land \neg x_{g,3} \land \neg x_{g,4}) \right) \implies \left( \bigvee_{g=1}^{3} (x_{g,3} \land \neg x_{g,1} \land \neg x_{g,2} \land \neg x_{g,4}) \right)$$
### (d) Cada gabinete só pode acomodar, no máximo, 2 pessoas.
**Intuição:** Esta é uma restrição de capacidade (*at-most-2*).
Dizer que um gabinete $g$ pode ter **no máximo 2 pessoas** é logicamente equivalente a dizer que **não pode conter 3 (ou mais) pessoas**.

Para cada gabinete $g$, qualquer subconjunto de 3 pessoas escolhidas de entre $\{A, N, P, M\}$ não pode estar simultaneamente em $g$. O número de combinações de 3 pessoas é $\binom{4}{3} = 4$:
1. $\{A, N, P\}$
2. $\{A, N, M\}$
3. $\{A, P, M\}$
4. $\{N, P, M\}$

Assim, para cada gabinete $g \in \{1, 2, 3\}$, temos a conjunção:
$$\bigwedge_{g=1}^{3} \left( \neg(x_{A,g} \land x_{N,g} \land x_{P,g}) \land \neg(x_{A,g} \land x_{N,g} \land x_{M,g}) \land \neg(x_{A,g} \land x_{P,g} \land x_{M,g}) \land \neg(x_{N,g} \land x_{P,g} \land x_{M,g}) \right)$$

Pelas leis de De Morgan, obtemos diretamente em CNF (4 cláusulas por gabinete, 12 cláusulas no total):
$$\bigwedge_{g=1}^{3} \begin{pmatrix}
(\neg x_{A,g} \lor \neg x_{N,g} \lor \neg x_{P,g}) \\
\land \\
(\neg x_{A,g} \lor \neg x_{N,g} \lor \neg x_{M,g}) \\
\land \\
(\neg x_{A,g} \lor \neg x_{P,g} \lor \neg x_{M,g}) \\
\land \\
(\neg x_{N,g} \lor \neg x_{P,g} \lor \neg x_{M,g})
\end{pmatrix}$$




Excelente esforço! Estás a pensar no problema com a atitude certa de engenharia. Reparei que adotaste a convenção de índices:
* $x_{g, p}$ onde o primeiro índice $g \in \{1, 2, 3\}$ é o **gabinete** e o segundo $p \in \{1, 2, 3, 4\}$ é a **pessoa** ($1 = \text{Ana}, 2 = \text{Nuno}, 3 = \text{Pedro}, 4 = \text{Maria}$). 

Não há qualquer problema com esta convenção, desde que sejamos consistentes. Vamos analisar criticamente o que fizeste, ponto por ponto.


### Parte 2: O SAT Solver e a Exploração de Soluções

1. **Como tirar partido:**
   Passamos a conjunção de todas as regras $\Phi = \phi_{(a)} \land \phi_{(b)} \land \phi_{(c)} \land \phi_{(d)}$ ao SAT solver. Se retornar **SAT**, o solver dá-nos uma valoração das variáveis $x_{g,p}$. As variáveis com valor `true` indicam quem fica em que gabinete.

2. **Há várias alternativas?**
   Com 4 pessoas e 3 gabinetes, o princípio do pombal diz-nos que a ocupação dos gabinetes será do tipo $(2, 2, 0)$ ou $(2, 1, 1)$. Existem várias permutações válidas compatíveis com as restrições.

3. **Como explorar todas as soluções (*Model Enumeration*):**
   Usamos a técnica de **Cláusulas de Bloqueio (*Blocking Clauses*)**:
   * O solver devolve uma primeira solução $M_1$.
   * Identificamos a atribuição concreta de pessoas a gabinetes em $M_1$, digamos $\bigwedge_{p=1}^4 x_{g_p, p}$.
   * Adicionamos a negação dessa solução à fórmula base:
     $$\Phi \leftarrow \Phi \land \left( \bigvee_{p=1}^4 \neg x_{g_p, p} \right)$$
   * Chamamos novamente o solver. Ele é obrigado a procurar uma distribuição diferente. Repetimos até o solver devolver **UNSAT**.

---

### Parte 3: Descrição Lógica das Novas Afirmações

#### (a) *"Se a Maria ocupar o gabinete um, então ela ficará sozinha nesse gabinete."*
* **Antecedente:** Maria ($p=4$) está no gabinete 1 $\rightarrow x_{1,4}$.
* **Consequente:** Ninguém mais está no gabinete 1 $\rightarrow \neg x_{1,1} \land \neg x_{1,2} \land \neg x_{1,3}$.
* **Fórmula ($\psi_a$):**
  $$x_{1,4} \implies (\neg x_{1,1} \land \neg x_{1,2} \land \neg x_{1,3})$$
  *(Em CNF: $(\neg x_{1,4} \lor \neg x_{1,1}) \land (\neg x_{1,4} \lor \neg x_{1,2}) \land (\neg x_{1,4} \lor \neg x_{1,3})$)*

#### (b) *"Se a Ana e o Nuno ficarem no mesmo gabinete, então a Maria e o Pedro terão que partilhar também um outro gabinete."*
* **Antecedente:** A Ana ($p=1$) e o Nuno ($p=2$) partilham um gabinete qualquer $g$:
  $$\bigvee_{g=1}^{3} (x_{g,1} \land x_{g,2})$$
* **Consequente:** A Maria ($p=4$) e o Pedro ($p=3$) partilham um gabinete (que terá de ser outro, pois cada gabinete só leva 2 pessoas):
  $$\bigvee_{k=1}^{3} (x_{k,3} \land x_{k,4})$$
* **Fórmula ($\psi_b$):**
  $$\left( \bigvee_{g=1}^{3} (x_{g,1} \land x_{g,2}) \right) \implies \left( \bigvee_{k=1}^{3} (x_{k,3} \land x_{k,4}) \right)$$

---

### Parte 4: Testar a Veracidade com o SAT Solver (Prova por Refutação)

Queremos saber se estas afirmações são propriedades invariantes do sistema — isto é, se são verdades absolutas decorrentes das regras originais ($\Phi \models \psi$).

Como um SAT solver decide **satisfiabilidade** e não diretamente validade, usamos a regra de ouro dos Métodos Formais: o **Princípio da Refutação**:
$$\Phi \models \psi \iff \Phi \land \neg \psi \text{ é INSATISFIÁVEL (UNSAT)}$$

Lembram-se que $\neg(P \implies Q) \equiv P \land \neg Q$?

#### Testar a afirmação 3.(a):
1. Construímos a negação de $\psi_a$:
   $$\neg \psi_a \equiv x_{1,4} \land (x_{1,1} \lor x_{1,2} \lor x_{1,3})$$
   *(Estamos a dizer ao solver: "Tenta encontrar uma distribuição válida onde a Maria esteja no gabinete 1 acompanhada por alguém".)*
2. Executamos o solver com $\Phi \land \neg \psi_a$.
3. **Resultado:**
   * Se der **UNSAT**: A afirmação é **VERDADEIRA**.
   * Se der **SAT**: A afirmação é **FALSA**, e o modelo devolvido mostra exatamente a Maria a partilhar o gabinete 1 com outra pessoa. *(Dica: de facto é SAT! Por exemplo, a Maria pode perfeitamente partilhar o gabinete 1 com a Ana, ficando o Pedro no 2 e o Nuno no 3)*.

#### Testar a afirmação 3.(b):
1. Construímos a negação de $\psi_b$:
   $$\neg \psi_b \equiv \left( \bigvee_{g=1}^{3} (x_{g,1} \land x_{g,2}) \right) \land \bigwedge_{k=1}^{3} (\neg x_{k,3} \lor \neg x_{k,4})$$
   *(Dizemos ao solver: "Encontra uma distribuição onde a Ana e o Nuno partilhem gabinete, mas o Pedro e a Maria NÃO partilhem gabinete nenhum".)*
2. Executamos o solver com $\Phi \land \neg \psi_b$.
3. **Resultado:**
   * Se der **UNSAT**: A afirmação é **VERDADEIRA**.
   * Se der **SAT**: A afirmação é **FALSA**. *(Dica: também dá SAT! Se a Ana e o Nuno estiverem juntos no gabinete 1, o Pedro pode ficar sozinho no gabinete 2 e a Maria sozinha no gabinete 3. Como a Ana não ficou sozinha, a regra (c) não é violada!)*
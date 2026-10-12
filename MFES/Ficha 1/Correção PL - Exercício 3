# Exercício 2 (Alocações de Aulas)

## 0. Domínio

Definimos claramente os nossos conjuntos:
* **Aulas (sequência temporal ordenada):** $A = \{1, 2, 3, 4, 5\}$
* **Formadores:** $\mathcal{F} = \{A, B, C\}$
  * $A \rightarrow \text{Ana}$
  * $B \rightarrow \text{Beatriz}$
  * $C \rightarrow \text{Carlos}$

**Variáveis Proposicionais:**
> Definimos $x_{a, f}$ como sendo **verdadeira** se e só se o formador $f$ leciona a aula $a$, onde $a \in A$ e $f \in \mathcal{F}$.

Temos um espaço de estados gerado por $5 \times 3 = 15$ variáveis booleanas.*

## 1. Restrições

### (a) O Carlos não pode dar a primeira aula
**Intuição:** Uma restrição unitária direta sobre a aula $1$ e o formador $C$:
$$\neg x_{1,C}$$

### (b) Se a Beatriz der a primeira aula, não poderá dar a última
**Intuição:** Uma implicação direta: se $x_{1, B}$ for verdade, então $x_{5, B}$ tem de ser falso.

$$x_{1, B} \implies \neg x_{5, B}$$

Na forma de cláusula (CNF):
$$\neg x_{1, B} \lor \neg x_{5, B}$$

*(Repara que isto é logicamente equivalente a dizer que a Beatriz não pode dar simultaneamente a primeira e a última aula: $\neg (x_{1,B} \land x_{5,B})$).

### (c) Cada aula tem pelo menos um formador
**Intuição:** Garantia de cobertura total do serviço. Nenhuma aula pode ficar deserta (restrição de existência):

$$\bigwedge_{a=1}^{5} (x_{a, A} \lor x_{a, B} \lor x_{a, C})$$

### (d) As quatro primeiras aulas têm no máximo um formador.
**Intuição:** Para as aulas $a \in \{1, 2, 3, 4\}$, não é permitida a copresença de formadores. Nenhum par de formadores distintos pode lecionar a mesma aula:

$$\bigwedge_{a=1}^{4} \left( \neg(x_{a, A} \land x_{a, B}) \land \neg(x_{a, A} \land x_{a, C}) \land \neg(x_{a, B} \land x_{a, C}) \right)$$

Pela lei de De Morgan, obtemos diretamente o formato de cláusulas:
$$\bigwedge_{a=1}^{4} \begin{pmatrix}
(\neg x_{a, A} \lor \neg x_{a, B}) \\
\land \\
(\neg x_{a, A} \lor \neg x_{a, C}) \\
\land \\
(\neg x_{a, B} \lor \neg x_{a, C})
\end{pmatrix}$$

*(Nota pedagógica: A conjunção da regra (c) com a regra (d) garante que nas quatro primeiras aulas temos **exatamente um** formador por aula).*

### (e) A última aula pode ter no máximo dois formadores
**Intuição:** Na aula $a = 5$, podem estar 1 ou 2 formadores. O que é que é **estritamente proibido**?
Proibido é estarem os **três formadores simultaneamente** na aula 5! 

Como só existem 3 formadores no total ($\{A, B, C\}$), dizer que *"no máximo 2 estão presentes"* é logicamente equivalente a negar que *"todos os 3 estejam presentes"*:

$$\neg (x_{5, A} \land x_{5, B} \land x_{5, C})$$

Aplicando a lei de De Morgan, obtemos uma cláusula muito simples:
$$\neg x_{5, A} \lor \neg x_{5, B} \lor \neg x_{5, C}$$

### (f) Nenhum formador pode dar 4 aulas consecutivas.
**Intuição:** Uma restrição sobre **janelas temporais deslizantes** (*sliding windows*). 
Quantas janelas de 4 aulas consecutivas existem num horizonte de 5 aulas? Apenas **duas**:
1. Janela 1: Aulas $\{1, 2, 3, 4\}$
2. Janela 2: Aulas $\{2, 3, 4, 5\}$

Para cada formador $f \in \{A, B, C\}$, ele não pode estar ativo nas quatro aulas da janela 1, **nem** nas quatro aulas da janela 2:

$$\bigwedge_{f \in \{A, B, C\}} \left( \neg(x_{1, f} \land x_{2, f} \land x_{3, f} \land x_{4, f}) \land \neg(x_{2, f} \land x_{3, f} \land x_{4, f} \land x_{5, f}) \right)$$

Passando para cláusulas (CNF):
$$\bigwedge_{f \in \{A, B, C\}} \begin{pmatrix}
(\neg x_{1, f} \lor \neg x_{2, f} \lor \neg x_{3, f} \lor \neg x_{4, f}) \\
\land \\
(\neg x_{2, f} \lor \neg x_{3, f} \lor \neg x_{4, f} \lor \neg x_{5, f})
\end{pmatrix}$$



## 2. SAT Solver

**Execução do SAT Solver**
Submetemos a teoria $\Phi$ ao solver. O algoritmo analisa o espaço de pesquisa combinatório ($2^{15} = 32.768$ configurações possíveis) e produz um de dois vereditos:

1. **`UNSAT` (Insatisfiável):**
   * Significa que as restrições são contraditórias e impossíveis de cumprir em simultâneo. 
   * Na prática, isto provaria matematicamente à direção do curso que é impossível montar o horário pretendido sem relaxar alguma das regras.
2. **`SAT` (Satisfiável):**
   * O solver encontrou uma solução consistente e devolve uma **valoração** (um *modelo* $M \models \Phi$), que consiste numa lista com a atribuição de verdade (`true`/`false`) para cada uma das 15 variáveis.

Para obter a escala final de aulas, basta fazer o processo inverso do mapeamento:

* Filtramos apenas as variáveis às quais o solver atribuiu o valor **`true`** (literais positivos).
* Se a variável correspondente a $x_{a, f}$ for verdadeira, alocamos o formador $f$ à aula $a$.

**Garantia formal:** Graças à correção da nossa modelação lógica:
* Asseguramos que as aulas 1 a 4 têm exatamente 1 formador (pela junção de (c) e (d)).
* Asseguramos que a aula 5 tem 1 ou 2 formadores (por (c) e (e)).
* Garantimos o cumprimento estrito de todos os períodos de descanso e restrições individuais ((a), (b) e (f)).

O resultado final é uma grelha temporal completa e **provadamente correta por construção**.


# Applied RL

Status: Not started

### Ideen Projekte

- Ensemble zur Messung der Unsicherheit (Gerrit & Jonas)
    1. **Kernkonzept & Idee**
        - **Paper-Problem:** KDE schätzt die Dichte im Zustandsraum $p(s)$. Es skaliert schlecht in höheren Dimensionen und bestraft datenarme Zustände auch dann, wenn deren Physik trivial vorhersehbar ist.
        - **Die Lösung:** Ersetzen von KDE durch ein **Dynamics-Ensemble**. Es misst die **epistemische Unsicherheit (Modellunsicherheit)** der Übergangsdynamik statt der reinen Zustandsdichte.
    2. **Funktionsweise**
        - **Model-Ensemble:** N LSTMs (z. B. N=5) werden mit unterschiedlicher Zufallsinitialisierung auf den Offline-Daten trainiert.
        - **Environment Step:** Der Mittelwert aller Modellprognosen dient als nächster Zustand $\hat{s}_{t+1}$.
        - **Reward Penalty:** Die Varianz $\sigma^2_{\text{ensemble}}(s,a)$der Prognosen dient als stetige Strafe im Reward: $\text{Reward} = \text{Standard-Reward} - \lambda \cdot \sigma^2_{\text{ensemble}}$
    3. **Dichte vs. Unsicherheit**
        - **Datendichte (KDE/GMM):** Misst $p(s)$ rückwärtsgewandt (reine Datenmenge im Raum).
        - **Unsicherheit (Ensemble):** Misst $\sigma^2(s,a)$ vorwärtsgewandt (Stabilität der Physik-Prognose).
        - **Vorteil:** Verhindert grundlose Bestrafungen bei wenigen Daten mit einfacher Physik und schützt vor chaotischen Bereichen trotz vieler Daten.
    4. **Anderes Environment** 
        - Um die Vorteile von Ensemble Modellen zu klarzumachen sind Umgebungen notwendig bei dem Unsicherheit ≠ Dichte ist
        - Unterschiedliche Dynamiken in System
            - Double Pendulum: dm_control/acrobot-swingup-v0”
- Lagrangian Constraint (Gelernter Multiplikator) (Maram & Theresa)
    1. Kernkonzept & Idee
        - **Paper-Problem:** Die Penalty p ist ein fester Hyperparameter. Bei zu kleinem p ignoriert der Agent die Zone (15% Restquote bei p=30). Bei zu großem p lernt der Agent gar nicht mehr (Reward-Signal wird von Penalty dominiert). Das optimale p hängt vom Environment ab und muss manuell getuned werden.
        - **Die Lösung:** Ersetzen der festen Penalty durch einen **gelernten Lagrange-Multiplikator α**. Das System formuliert Zone-Vermeidung als Constraint und lernt automatisch, wie stark bestraft werden muss.
    2. Funktionsweise
        - **Constrained MDP:** Die Optimierung wird umformuliert als:
        
        ```
        max_π J(π)   s.t.   C(π) = E[Σ_t c(s_t)] ≤ ε
        ```
        
        wobei c(s_t) = 1[s_t ∈ Zone] (oder c(s_t) = 1 - density(s_t) für stetige Variante).
        
        - **Lagrangian Relaxation:** Umwandlung in ein unrestringiertes Sattelpunkt-Problem:
        
        ```
        min_{α≥0} max_π  J(π) - α · (C(π) - ε)
        ```
        
        - **Dual Update:** α wird mit eigenem Optimizer (Adam) aktualisiert:
        
        ```
        α ← max(0, α + η_α · (C(π) - ε))
        ```
        
        Wenn Agent zu oft in Zone → α steigt. Wenn Constraint erfüllt → α sinkt.
        
        - **Effektiver Reward pro Step:**
        
        ```
        Reward = Standard-Reward - α · c(s_t)
        ```
        
        Sieht aus wie das Paper, aber α ist keine Konstante sondern ein gelernter Parameter.
        
    3. Feste Penalty vs. Lagrangian
        - **Feste Penalty (Paper):** Manuelles p, keine Garantie dass Constraint erfüllt wird, Trade-off zwischen Task-Performance und Vermeidung ist statisch.
        - **Lagrangian:** α adaptiert sich, konvergiert gegen das minimale α das die Constraint-Verletzung ≤ ε hält. Kein manuelles Tuning von p nötig — einziger Hyperparameter ist ε (erlaubte Verletzungsrate, z.B. 2%).
        - **Vorteil:** Löst das 15%-Problem strukturell — α wird so lange hochgefahren bis Zone-Besuche ≤ ε. Geschlossener Regelkreis statt offene Steuerung.
    4. Environment-Anforderungen
        - Funktioniert auf dem **Standard-Pendulum** (gleicher Task wie Paper) — kein Environment-Wechsel nötig.
        - Vorteil zeigt sich besonders bei Environments wo das optimale p unklar ist oder sich über das Training ändert.
        - Kombinierbar mit beliebigem Cost-Signal: binäre Zone, KDE-Dichte, oder Ensemble-Varianz als c(s_t).
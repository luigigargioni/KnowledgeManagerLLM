# TODO


1. Creare template per risposte dei tool call..per molte di esse non c'è bisogno che la risposta passi nuovamente dal modello basta avere un qualche testo strutturato
    - Cosa facciamo se l'llm si sente di chiamare più di un tool? Imponiamo che sia sempre soltanto uno oppure il template permette più output?
2. Valutare l'utilizzo del formato TOON [libreria qui](https://github.com/toon-format/toon-python) per velocizzare l'analisi dei JSON da parte dell'llm
3. Creare file di mock up con i dati che potrebbero successivamente essere ottenuti dal database vettoriale 
4. [FATTO] Scrivere tool deterministici per l'identificazione dei conflitti
5. [FATTO] Incorporare un messaggio tool/user all'inizio per passare dati rilevanti, es data e ora, senza che ci sia bisogno di chiamare un tool apposito



# Note / da discutere
- Se un farmaco richiede di essere assunto a stomaco pieno e supponiamo che ci sia un conflitto per l'orario che ha scelto l'utente. Ha senso che l'algoritmo deterministico suggerisca di anticipare prima di pranzo? L'LLM dovrebbe quindi combinare sia il check semantico che quello deterministico per risolvere i conflitti?
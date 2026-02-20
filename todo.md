# TODO


1. Valutare l'utilizzo del formato TOON [libreria qui](https://github.com/toon-format/toon-python) per velocizzare l'analisi dei JSON da parte dell'llm --> Per ora niente: leggendo meglio, essendo uno standard nuovo i modelli non sono ancora allenati e quindi bisogna istruirli. Nel nostro caso il JSON è molto corto quindi il risparmio sarebbe minimo. Ad ora il gioco non vale la candela.



# Note / da discutere
- Se un farmaco richiede di essere assunto a stomaco pieno e supponiamo che ci sia un conflitto per l'orario che ha scelto l'utente. Ha senso che l'algoritmo deterministico suggerisca di anticipare prima di pranzo? L'LLM dovrebbe quindi combinare sia il check semantico che quello deterministico per risolvere i conflitti? (possibile limitazione o future work)
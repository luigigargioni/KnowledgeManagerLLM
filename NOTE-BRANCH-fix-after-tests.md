# Branch `fix-after-tests` — note di consegna

Branch staccato da `dev`. Contiene **11 commit**, 34 file toccati, ~3200 righe aggiunte e ~1170 rimosse.

Il lavoro si divide in tre blocchi, che conviene leggere in quest'ordine perché ognuno dipende dal precedente:

1. **Infrastruttura LLM** — client unico con rate limiting, due ruoli configurabili separatamente, parametri di sampling.
2. **Harness di valutazione** — la valutazione passa dal transcript al diff programmatico dei dati.
3. **Correzione bug** — analisi statica del repo dopo i primi test, più i difetti emersi eseguendo gli scenari.

> **Prerequisito di lettura:** `CLAUDE.md` è stato scritto su questo branch e documenta l'architettura in dettaglio (ruoli LLM, struttura multi-agente, dove vanno i token, soglie RAG, gotcha). Questo documento **non** lo duplica: racconta *cosa è cambiato e perché*, e rimanda lì per il *come funziona*.

---

## 1. Infrastruttura LLM

### `src/llm_client.py` (nuovo, ~470 righe)

Unico punto in cui viene costruito un client. Prima ogni modulo si creava il suo.

- **Due ruoli indipendenti**: `MAIN_LLM` (il sistema sotto test: therapy manager, checker, extractor) e `SIM_LLM` (l'harness: caregiver simulato e judge). Ognuno con **provider e modello propri**, così un modello locale può essere valutato da uno cloud. Ogni `SIM_*` non impostato eredita dal corrispettivo `MAIN`.
- **Rate limiting client-side** su finestra scorrevole di 60s, applicato *prima* di inviare. Le quote sono per `provider:model`, perché è così che le contano i provider. Nato per il tier gratuito Groq (8K token/minuto), dove un batch non regolato passa il tempo a rimbalzare sui 429.
- **Stima dei token auto-calibrante**: il rapporto caratteri→token viene misurato per quota sulle prime risposte reali. Questi prompt tokenizzano intorno ai 7.5 char/token e una stima ingenua dimezzava il throughput.
- **Due fallimenti resi espliciti** invece di essere ritentati alla cieca:
  - `DailyQuotaExceeded` — quota giornaliera esaurita, o il provider chiede un'attesa più lunga di `LLM_MAX_RETRY_WAIT`. `test.py` lo intercetta e ferma il batch, conservando gli scenari già valutati.
  - `RequestTooLarge` — HTTP 413, il prompt da solo supera il budget al minuto: nessuna attesa può aiutare.
- **Parametri rifiutati**: `reasoning_effort` viene scartato e la chiamata ritentata se il modello risponde 400 (una richiesta sprecata per modello per processo, con warning nel log). Serve a non dover accordare la manopola al modello quando si cambia provider.

### Parametri di sampling — `TEMPERATURE` e `SEED`

Aggiunti in `LLMConfig` e inviati da `llm_client`. **Convenzione a tre stati**, la stessa già usata da `REASONING_EFFORT`:

| in `.env` | effetto |
|---|---|
| variabile assente | eredita dal ruolo MAIN (`None` per MAIN stesso) |
| presente ma vuota | **non inviare** il parametro, nessuna ereditarietà |
| presente con valore | quel valore |

A campi vuoti i parametri non finiscono affatto nella richiesta, quindi vale il default del provider: **il comportamento è identico a prima**. Un valore malformato fallisce al caricamento con un messaggio che nomina la variabile (`SEED='xyz' in .env is not a valid int`), non a metà batch.

Tre scelte da conoscere:

- **`temperature` e `seed` NON sono droppabili.** Il meccanismo che scarta i parametri rifiutati vale solo per `reasoning_effort`. Scartare un parametro di sampling farebbe finire il batch campionando diversamente da come la configurazione dichiara, e i risultati verrebbero attribuiti a impostazioni mai in vigore. Chi li rifiuta deve far fallire la run.
- **Il log di sessione registra il sampling** usato, incluso `temperature=provider default` — perché "default" è una scelta il cui valore dipende dal backend, e va scritta se i risultati devono essere confrontabili tra run.
- **La scheda del modello gpt-oss raccomanda `temperature=1.0`** con `top_p=1.0` (<https://github.com/openai/gpt-oss>). Abbassarla per ridurre la varianza è una deviazione deliberata dal punto operativo raccomandato, da misurare contro la qualità delle risposte.

**Su quale ruolo mettere il seed.** I due ruoli vogliono politiche opposte:

- **Judge (`SIM_SEED`) → fissalo.** La valutazione dovrebbe essere una funzione deterministica del transcript. Misurato: sullo stesso input byte per byte il judge ha dato `completed×3/partial×2` in un campione e `partial×5/completed×1` in un altro. Un seed fisso rende confrontabile una rivalutazione, senza contropartite.
- **Therapy manager (`SEED`) → attenzione.** Qui si misura la *capacità* dell'agente. Con un seed fisso non si misura quanto è bravo, si misura **una singola estrazione**: uno scenario sfortunato lo resterà a ogni riesecuzione, e uno fortunato sembrerà solido. Per i numeri da pubblicare servono N run con seed diversi, riportando media e dispersione.

---

## 2. Harness di valutazione

### La valutazione è basata sul diff, non sul transcript

`src/therapy_diff.py` (nuovo) calcola in codice la differenza fra la terapia iniziale e quella finale. Il judge riceve quel diff e il suo prompt dichiara che **il diff, non il transcript, è l'autorità**.

Esiste perché l'assistente dichiarava con regolarità modifiche che non aveva mai applicato, e un transcript con una conferma ben formattata è indistinguibile da un successo reale. **Non "semplificare" il judge facendogli leggere solo il transcript.**

L'eccezione voluta: un ramo condizionale può prescrivere di *non* agire ("do not proceed with it"), e lì lo stato atteso è un diff vuoto con obiettivo `completed`. Le due regole sono in tensione per costruzione e il prompt le dichiara entrambe.

### Lo script del caregiver viene trattenuto in parte

`scenario_loader.split_objectives()` sottrae al caregiver il titolo dello scenario, il preambolo e le clausole condizionali (`If the assistant…`, `Verify that the assistant…`). **71 scenari su 105** hanno una clausola simile, che di norma fissa lo stato finale atteso: consegnarla in anticipo significa dire la risposta al caregiver, che poi solleva il problema per primo o reagisce a un warning mai arrivato.

`test.py` consegna la parte trattenuta **sull'evento, non al turno N**: solo quando l'assistente ha davvero sollevato il punto. Se non lo fa mai, il caregiver non lo saprà e `evaluation["branch_exercised"]` registra `False`.

### Il gate scatta su una cosa sola: il sistema ha bloccato l'azione

`Chat._record_issue_signals` ispeziona ogni risultato di tool e registra il campo `issue` che `tools.py` attacca ai fallimenti bloccanti (`schedule_conflict`, `missing_dependency`, `temporal_ordering`, `dependency_blocked`), più il marker di medicina non trovata.

**Tre trigger più deboli sono stati provati e misurati fuori — non reintrodurli:**

| trigger scartato | perché |
|---|---|
| keyword nella risposta | scatta sulla parola "warning" in *"no conflicts were reported, but there was a history warning"*. Nello scenario 17 ha consegnato le istruzioni un turno in anticipo |
| verdetto del checker | il formato tiene (64/64 parsate) ma il checker commenta tutto, quindi l'assenza di `NO_CONFLICTS` copre anche osservazioni di qualità |
| hit sulla storia paziente | ne scatta uno praticamente a ogni richiesta, mentre la scrittura va comunque a buon fine |

Gli ultimi due condividono ciò che li squalifica: **l'attività viene scritta lo stesso**. Il sistema ha osservato qualcosa, non ha fermato niente.

Il costo è noto e va tenuto presente leggendo i risultati: dei 71 scenari condizionali, circa 30 hanno un trigger di forma bloccante e ricevono la clausola di reazione. Gli altri girano e vengono valutati sulla condotta, ma il caregiver improvvisa e il loro stato finale è meno determinato.

---

## 3. Bug corretti

Tutti verificati eseguendo il codice, non per lettura. I bug erano presenti anche nel `src/` committato (verificato contro `git show HEAD:src/tools.py`).

### Bloccanti

| # | Dove | Problema | Verifica |
|---|---|---|---|
| 1 | `sql_db.py` `seed_test_data` | `therapy` assegnata solo dentro il `try`: senza `data/patients/<id>/therapy.json` — **che non esiste per nessuno dei 9 pazienti** — la riga `if therapy:` sollevava `UnboundLocalError`. Chiamata da `main.py` e dalla UI Streamlit appena Postgres risponde | ora ritorna `{"status": "skipped"}` |
| 2 | `tools.py` `update_therapy_activity` | Il controllo di ordinamento temporale era dentro `if new_deps is not None`: cambiando **solo l'orario** il vincolo non veniva riverificato. Spostare un farmaco prima della sua dipendenza passava con `success` | ora `temporal_ordering`, file invariato |
| 3 | `chat.py` `execute_tool` | Doppio encoding JSON nella delega: `tc.function.arguments` è già una stringa e `_send_to_agent` ci faceva `json.dumps`. Il checker riceveva `"{\"message\": \"…\"}"` | ora riceve il dict |
| 4 | `tools.py` `remove_therapy_activity` | Il messaggio di rimozione bloccata mostrava gli **activity_id** (`Cannot remove 'ml_001' because it is a dependency of: med_001`), che il prompt del manager vieta di mostrare al caregiver. Ed è un `issue` bloccante, quindi viene rilanciato quasi sempre | ora usa i nomi |
| 5 | `tools.py` `update_therapy_activity` | Un update senza campi rispondeva `successfully updated` — un falso positivo generato dal tool, cioè dalla fonte che il prompt indica come unica autorità | ora `error` esplicito |

### Correttezza e robustezza

| Dove | Problema |
|---|---|
| `test.py` | Il default di `--to` contava i file `.json` (105) invece degli id numerati (1–100): un batch completo chiedeva 5 scenari inesistenti e non eseguiva mai gli `example*.json` |
| `main.py` | `recursion_limit=30` in LangGraph conta i *super-step*, quindi ~15 turni contro i 30 di `test.py`; e `GraphRecursionError` non era catturato, uccidendo la run **prima** della valutazione. Ora derivato da `--max-turns` e intercettato |
| `tools.py` `_save_therapy` | Scrittura non atomica (`open("w")` tronca subito). Ora `.tmp` + `os.replace`: un errore lascia il file precedente intatto |
| `tools.py` | `valid_from`/`valid_until` a stringa vuota salvati alla lettera (la normalizzazione era su copie locali). Ora `None` |
| `chat.py` | Snapshot della terapia salvato anche quando il tool **falliva**, creando duplicati identici |
| `chat_interface.py` | Il rewind usava `list.index(message)`: i dict si confrontano per valore, quindi due turni identici ("sì", "ok") riportavano al punto sbagliato |
| `chat_interface.py` | "Past Sessions" cercava in `logs/<patient_id>/` mentre il logger scrive in `logs/<timestamp>/`: la feature non trovava mai nulla |
| `utils.py` | `[CHAT][ISSUE]` finiva in `chat.log` (il filtro accettava ogni riga con prefisso `[CHAT]`) ma `parse_chat_log` non lo riconosce come turno e lo attaccava al messaggio **precedente** — 3 turni su 28 corrotti in un solo scenario. Colpiva il "Load session" di Streamlit |
| `config_loader.py` | Password non url-encoded nella connection string: `@`, `:`, `/` la rompevano |
| `sql_db.py` `load_session` | `x["valid_until"]` invece di `.get()`: un'attività senza la chiave faceva fallire il caricamento |
| `results_extractor.py` | Nessun limite sulla lunghezza delle celle: superare i 32767 caratteri di Excel avrebbe reso illeggibile l'intero workbook cumulativo |
| `session_extractor.py` | Nuovo client `OpenAI` (nuovo pool HTTP) a ogni chiamata |
| `utils.py` | `StartWithFilter` non chiamava `super().__init__()` |

### Emersi eseguendo gli scenari

**`duration_minutes` — messaggio d'errore fuorviante.** Il modello mandava `duration_minutes: 0` (ragionevole: prendere una pillola è istantaneo) e riceveva `must be a positive integer (e.g. 30, not 30.0)` — una lamentela sul **formato**. Correggeva quindi il formato scegliendo `1`, e il farmaco da 1 minuto veniva piazzato alle 12:29 per finire esattamente all'inizio del pranzo. Ora i due casi hanno messaggi distinti, e la descrizione nello schema del tool previene l'errore a monte.

**Judge — regola sui conflitti non applicata.** Il prompt diceva già che spostare l'orario per risolvere un conflitto va valutato `completed`, ma la regola stava in `## Notes` mentre la regola severa che la contraddiceva stava nella sezione dell'autorità. Spostata dove serve. Misurato su 11 estrazioni a input congelato: `completed` 4/11 → **11/11**, con gli altri scenari invariati.

---

## 4. Risultati dei test

Due batch degli scenari 1–10 (OpenAI / `gpt-5.4-mini` su entrambi i ruoli):

| | prima | dopo |
|---|---|---|
| completed / partial / failed | 7 / 3 / 0 | 8 / 1 / 1 |
| obiettivi | 10/13 (76.9%) | 11/13 (84.6%) |
| errori | 0 | 0 |

**Attenzione a leggere questa tabella come un miglioramento netto.** Su 4 scenari che hanno cambiato stato, verificando a input congelato è risultato che almeno uno (il 7) era **varianza della conversazione**, non effetto delle correzioni. Con due LLM in serie e temperatura al default, il rumore fra due run identiche è dell'ordine di grandezza degli effetti cercati. Serviranno più seed per numeri difendibili.

**Condotta del therapy agent**, misurata deterministicamente dai log su 20 esecuzioni di scenario:

| controllo | esito |
|---|---|
| activity_id mostrati al caregiver | **0** |
| scenari che modificano la terapia senza safety check preventivo | **0/10** |
| scenari che dichiarano modifiche con diff vuoto | **0** |
| loop agente esaurito (10 iterazioni) | **0** |
| errori tool auto-inflitti | 2 su 106 chiamate (1.9%) |

---

## 5. Come far girare

Tutti i comandi da `src/` (gli import sono piatti: dalla root del repo falliscono).

```bash
pip install -e .          # dalla root; richiede Python >= 3.14
cd src

python main.py                        # chat interattiva
python main.py -i <script.md>         # modalità agente
streamlit run chat_interface.py       # UI web

python test.py                        # batch: tutti gli scenari numerati
python test.py --from 1 --to 10       # sottoinsieme
ruff check . && ruff format .
```

`test.py` gira senza database. `main.py` e la UI Streamlit richiedono PostgreSQL. Non esiste una suite di unit test: "far girare i test" significa eseguire gli scenari e leggere i verdetti del `JudgeAgent`, il che richiede un backend LLM vivo e alcuni minuti per scenario.

**Per passare a Ollama** vanno cambiati **due** valori, non uno: `PROVIDER=ollama` **e** `MODEL=gpt-oss:20b` (più i corrispettivi `SIM_`). Solo `PROVIDER` lascerebbe `MODEL=gpt-5.4-mini`, che verrebbe inviato a Ollama. In alternativa lasciare `MODEL` vuoto: il default per ollama è già `gpt-oss:20b`.

> **Nota sull'installazione:** se `import tools` da fuori `src/` risolve a `.venv/Lib/site-packages/tools.py`, l'installazione è una **copia** e non un link — succede se è stato eseguito `pip install .` senza `-e`. Si sistema con `pip uninstall KnowledgeManagerLLM && pip install -e .`. Con una copia vecchia in mezzo si finisce per testare codice morto senza accorgersene.

---

## 6. Punti aperti

**Da verificare su Ollama (non fatto, macchina non disponibile).** Su `gpt-5.4-mini` via API OpenAI, `temperature` e `reasoning_effort` sono **mutuamente esclusivi**: con `reasoning_effort` presente, l'API accetta solo `temperature=1` (400 riproducibile 3/3). Ollama documenta `temperature`, `seed` e `reasoning_effort` come supportati insieme (<https://docs.ollama.com/api/openai-compatibility>), quindi il vincolo dovrebbe essere una policy dell'API OpenAI e non una proprietà di gpt-oss — **ma non è stato confermato empiricamente**. C'è una nota su `LLMConfig` che lo ricorda. Da verificare anche che il `seed` venga effettivamente onorato, e la scala della temperatura (range 0.0–2.0 OpenAI contro 0.0–1.0 di alcuni modelli locali, `ollama/ollama#3151`).

**Modifica non richiesta alla terapia, non segnalata.** In uno scenario su 20 l'assistente ha spostato il **pranzo del paziente** di 45 minuti senza che nessuno lo chiedesse, e non è mai stato rimesso a posto. Aveva perso l'`add` di un farmaco (fallito, quindi senza id) e ha chiamato `update` con l'id della dipendenza. Da notare che le regole di condotta hanno tenuto — ha chiesto conferma prima di agire e ha **riportato onestamente** l'errore (*"The update did not affect Ibuprofen. It changed Lunch to 13:15 instead"*) — ma il danno è rimasto e il judge ha dato 3/3. Un controllo deterministico che confronti le attività toccate dal diff con quelle nominate dal caregiver, esposto come campo in `evaluation.json`, lo renderebbe visibile a chi rilegge i risultati. Non implementato.

**`results.xlsx` (276 KB) è tracciato nella root del repo** e non è in `.gitignore`. È output di test: lo stesso motivo per cui `data/therapy.json` è stato tolto dal tracking nel commit `389c018`. Da valutare se rimuoverlo.

**Varianza del judge non caratterizzata.** Il prompt del judge è stato modificato e l'effetto è stato misurato su 4 scenari con 11 estrazioni. L'effetto sugli altri 90 non è misurato. Per una risposta solida servirebbero ~30 estrazioni per cella su un campione di scenari con e senza conflitti — con 6 non si distingue nulla, verificato sul campo.

**Metodo, per chi continuerà.** Le correzioni al codice si verificano in modo deterministico e sono chiuse. Le modifiche ai **prompt** no: qualsiasi futura modifica va misurata su input congelati e con almeno una replica, altrimenti il rumore della pipeline è più grande dell'effetto cercato. In questa sessione un primo campione da 5 ha portato a una conclusione che il secondo ha smentito.

---

## Elenco commit

| commit | contenuto |
|---|---|
| `a357d00` | Refactor gestione terapia e vector DB; nasce `CLAUDE.md` |
| `dd42b5b` | Tracciamento terapia iniziale, logging risultati su Excel |
| `2979830` | Il caregiver non nomina più gli activity_id |
| `cd75636` | `split_objectives`: le clausole condizionali vengono trattenute |
| `104a391` | `llm_client.py`: client con rate limiting, due ruoli |
| `eff9946` | `_record_issue_signals`: il gate sui segnali bloccanti |
| `f6a4678` | Judge: rami che prescrivono di non agire; soglie RAG |
| `389c018` | `therapy.example.json`, `therapy.json` tolto dal tracking |
| `7165085` | Correzione bug (analisi statica) — 11 file |
| `be2837f` | `duration_minutes`, regola conflitti judge, filtro `chat.log` |
| `2b9479a` | `TEMPERATURE` e `SEED` configurabili, sampling nel log |

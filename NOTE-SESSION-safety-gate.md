# Sessione 2026-08-24 — gate di sicurezza, valutazione, backend OpenRouter

Modifiche **non ancora committate** sul branch `fix-after-tests` (ultimo commit: `818487c`).
17 file modificati, 5 nuovi, ~955 righe aggiunte.

Punto di partenza: `all_results_test_50.xlsx`, i risultati degli scenari 1–50 girati su
`ollama` + `gpt-oss-20b`. 27 `completed`, 7 `partial`, 16 `failed`.

Questo documento racconta **cosa è cambiato e perché**. `CLAUDE.md` è stato aggiornato in
parallelo e descrive *come funziona* il risultato: le due letture non si duplicano.

---

## 0. Cosa diceva l'analisi dei 50 scenari

Le 23 non-completed non erano 23 problemi dello stesso tipo. Distribuzione per causa radice:

| causa | scenari | n |
|---|---|---|
| il manager si auto-veta su un finding non bloccante | 42, 45, 46 | 3 |
| warning soft sollevato → il gate non lo vede → il caregiver improvvisa → il judge boccia | 3, 13, 14, 34 | 4 |
| warning non sollevato affatto (varianza) → branch non esercitato | 6, 15, 43, 44, 47 | 5 |
| **safety check mai delegato** | 32, 37 | 2 |
| **modifica dichiarata e mai applicata** (2ª operazione di una sequenza) | 18, 48 | 2 |
| `medicine_not_found` trattato come blocco duro → stallo | 4, 8 | 2 |
| scrittura sbagliata (orario/dipendenza) | 10 | 1 |
| il checker non produce la cautela di dominio | 36 | 1 |
| over-block su richiesta non condizionale + caregiver che capitola | 49 | 1 |
| clausola di scenario ambigua | 26 | 1 |
| obiettivo soppiantato da uno successivo vs valutazione a diff | 30 | 1 |

Due conclusioni contavano più delle altre.

**Il judge violava il proprio rubric su 6 scenari.** `judge_agent.py` diceva già che un branch
mai innescato è `partial` e che il caregiver non va mai valutato. Su 3, 13, 14, 34, 42 e 46 lo
ha bocciato per quello che il caregiver aveva detto o rifiutato — motivazione letterale sullo
scenario 14: *"the caretaker approved addition despite the risk"*. Con `SIM_SEED` fisso questi
errori si riproducono, non si mediano.

**Il gate di consegna non vedeva il trigger di 18 scenari su 33.** Classificando i 33 scenari
condizionali per il trigger che *intendevano* produrre:

| trigger previsto | consegnate |
|---|---|
| `schedule_conflict` | 5/6 |
| `dependency_blocked` | 2/2 |
| `temporal_ordering` | 1/2 |
| `missing_dependency` | 1/1 |
| `medicine_not_found` | 1/1 |
| **warning soft** (storia paziente / verdetto checker) | **6/18** |

E quei 6 non erano veri positivi: 4, 33, 35, 36, 43, 47 sono scattati tutti su un blocco di
scheduling o dipendenza estraneo allo scenario. Sul trigger soft il tasso di veri positivi era
**0/18**. Si vedeva in chiaro come contraddizione nel report: 36, 43 e 47 avevano
`branch_outcome=exercised` accanto a una nota del judge che diceva che il branch non era stato
esercitato.

---

## 1. Il gate di sicurezza — `src/safety.py` (nuovo, 260 righe)

Il verdetto del checker era una lista piatta di stringhe: una controindicazione assoluta e
*"12:45 is around lunch, so it is not fasting"* avevano la stessa forma. È per questo che
usarlo come segnale era stato scartato, e la ragione era buona. Ma il difetto non era il
verdetto: era l'assenza di una severità.

`check_result` diventa `[{"severity": …, "finding": …}]`, con la severità tagliata su **chi
deve decidere**:

- **`blocking`** — l'attività non deve esistere come richiesta. Nessuno nella conversazione può
  autorizzarla. **Rifiuta la scrittura.**
- **`caution`** — un rischio reale che il *caregiver* può accettare. **Non rifiuta la
  scrittura**: riportarlo e chiedere è dovere dell'assistente (vedi §1.2).
- **`remark`** — osservazione senza decisione attaccata. Non blocca e non segnala. È il livello
  che assorbe tutto ciò che prima produceva falsi positivi, ed è la ragione per cui il verdetto
  è utilizzabile adesso.

`blocking` e `caution` emettono un `issue` nel namespace di `tools.py`, quindi il gate di
consegna che già capiva una scrittura rifiutata li capisce senza modifiche.

### 1.1 Il check non è più opzionale — `Chat._enforce_safety_gate`

Sta davanti a `add`/`update`/`remove_therapy_activity`, **prima** di `tools.py`, nello stesso
imbuto in cui passano già le tool call di ogni agente. Tre rifiuti:

| rifiuto | quando | come si sblocca |
|---|---|---|
| `safety_check_required` | il checker non è mai stato interpellato su questa attività | chiama il checker, richiama la write — stesso turno, nessun turno di caregiver speso |
| `safety_blocked` | il verdetto corrente è `blocking` | non si sblocca: la via d'uscita è un'altra attività |

Esiste perché la regola nel prompt non bastava: su 6 scenari della stessa classe (aggiungere un
farmaco controindicato) il manager ha delegato al checker 4 volte e l'ha saltato 2, scrivendo il
farmaco senza alcun controllo (32, 37). Stesso prompt, stessa richiesta: la differenza era il
campionamento. È la stessa mossa con cui lo scheduling è stato tolto all'LLM e dato a `tools.py`.

`safety_check_required` **non** viene registrato come segnale di consegna: è uno scivolone
meccanico auto-correggibile, e registrarlo avrebbe aperto il gate su un *"anything else?"*
qualunque, il cui punto di domanda basta a `assistant_handed_back`. Finisce in un contatore
separato, `safety_checks_skipped`.

### 1.2 Il latch della `caution`, e perché è stato rimosso

Nella prima versione anche `caution` rifiutava la scrittura, una volta per attività, con
rilascio al turno successivo del caregiver — così la decisione gli arrivava per costruzione.

**Misurato su `gpt-oss-20b` a reasoning effort `low`, scenari 3, 13, 14: il modello leggeva il
rifiuto, lo riportava, e non richiamava più il tool.** Il caregiver diceva "procedi comunque" e
non veniva scritto niente. Il meccanismo aveva migliorato l'*osservabilità* del branch e
peggiorato l'esito reale.

Era anche la regola sbagliata nel merito: una `caution` è per definizione un rischio relativo
che il caregiver può accettare, e rifiutare una scrittura legittima è l'assistente che si arroga
una decisione — esattamente quello che il suo prompt gli vieta. Riportare e chiedere è tutto il
dovere, e quel dovere sta nel prompt, non in un latch nel codice.

Rimossi `_safety_cautions`, `_latched_caution_turn`, la logica di rilascio sul turno e il ramo
corrispondente nel gate. Il segnale continua a scattare perché nasce dal verdetto del checker,
che arriva *prima* della scrittura: la consegna della clausola è ora un effetto collaterale di
come funziona il sistema, non il motivo per cui il sistema è fatto così.

> **Non reintrodurlo.** Un assistente che scrive senza chiedere è un difetto del prompt da
> misurare, non un lucchetto da aggiungere.

### 1.3 Due decisioni interne, con la misura che le ha prodotte

- **L'autorità è il verdetto più recente, mai uno storico.** Un `blocking` era latched in modo
  permanente — che sembra giusto, una controindicazione assoluta non si dissolve a metà
  conversazione — e al primo run live ha bandito il **Paracetamolo** per il resto dello scenario
  32 sulla base di un finding che diceva *"contraindicated only if severe hepatic
  insufficiency"*, condizione che quel paziente non ha. Il Paracetamolo era la risposta giusta a
  quello scenario. Una classificazione sbagliata è più probabile di una controindicazione che
  cambia, quindi vince il giudizio più nuovo; la colonna `safety_verdicts` stampa tutti i
  verdetti con turno e severità, ed è ciò che rende visibile un modello che va a caccia di una
  risposta più morbida.
- **I verdetti hanno scope di sessione, non di turno.** Richiedere il check nello stesso turno
  della scrittura obbligava a ri-controllare prima di ogni write: lo scenario 32 ha speso 8
  turni, 50 richieste e 172K token a ri-controllare la stessa aspirina, poi ha esaurito i turni.
  Con il verdetto persistente: 5 turni, 20 richieste, 56K token, `completed`.
- **Il fallimento di formato è fail-open, ma rumoroso.** Un verdetto non parsabile o non
  tipizzato conta come controllato-senza-findings, viene loggato e finisce in
  `safety_verdicts_unparsed`. Fallire chiuso è il default giusto per un sistema clinico e quello
  sbagliato per un harness: bloccherebbe uno scenario per uno scivolone di formato e lo
  addebiterebbe al comportamento sotto test.

### 1.3 bis — il tasso di fallimento del formato, misurato

Sul batch dei 23 scenari a `low`: **4 su 23 (~17%)** hanno prodotto almeno un verdetto non
parsato o non tipizzato (4, 13, 32, 37). È un numero da conoscere prima di fidarsi di un run a
`low` — a `high` non si è presentato.

---

## 2. Il prompt del manager — la decisione è del caregiver

Nuova sezione in `agents/therapy_manager_agent.py`. Il buco era che la regola "restituisci
sempre la decisione" esisteva solo per i conflitti di *scheduling*, non per i finding di
sicurezza. Risultato, scenario 45: *"Because of this conflict, I cannot add the activity."* — e
46: *"Before proceeding, please discuss this precaution with the prescribing clinician."* In
entrambi i casi il tool non aveva bloccato nulla: era il modello che chiudeva la decisione.

Le regole aggiunte:

- una `caution` non ferma niente, e proprio per questo il dovere è tuo: dillo e chiedi **prima**
  di scrivere; scrivere e poi menzionare il rischio non è chiedere;
- `safety_blocked` significa che l'attività come richiesta non ci sarà: dillo, poi proponi
  un'alternativa o chiedi — e in ogni caso tieni aperta la conversazione;
- mai chiudere il turno rimandando a un clinico *invece* di chiedere al caregiver;
- mai abbandonare o ridurre una richiesta perché sembra rischiosa (scenario 49: 45min Lun/Mer/Ven
  diventati 30min Lun/Mer senza che nessuno lo chiedesse).

E una sezione sulle **alternative farmacologiche**, per lo stallo dello scenario 4 (13 turni,
Atenololo riproposto tre volte dopo averlo già dichiarato assente): nominare un farmaco solo
dopo che il checker ne ha confermato la presenza in KB; un farmaco dato per assente è fuori
gioco per il resto della conversazione; quando niente in KB va bene, dirlo e chiedere invece di
continuare a proporre nomi.

Lato checker, `agents/check_agent.py`:

- classificare `blocking` solo contro una condizione che *questo* paziente ha davvero — con
  l'esempio del Paracetamolo, perché è l'errore che è stato osservato;
- quando si riporta una controindicazione, guardare gli altri documenti che la lookup ha
  restituito: `get_medicine_data` ne torna diversi e un'alternativa della stessa classe è spesso
  fra quelli (verificato: una query su "Propranolol" restituisce anche atenololo e bisoprololo);
- il passo sul farmaco riguarda il farmaco che l'attività *introduce*: un farmaco già in terapia
  non richiede una nuova lookup solo perché la nuova attività lo segue.

---

## 3. Valutazione — smettere di valutare ciò che non è stato consegnato

### 3.1 Il judge sa se la clausola è arrivata

`test.py` passava al judge lo script **completo**, clausole condizionali incluse, anche quando
`split_objectives` le aveva trattenute e non erano mai state consegnate. Il judge valutava il
caregiver contro istruzioni che il caregiver non aveva mai ricevuto, e `branch_exercised` veniva
calcolato *dopo* la valutazione, quindi non poteva nemmeno saperlo.

Ora `JudgeAgent.evaluate` riceve `script` (la parte imperativa), `conditional_clause` e
`clause_delivered`. Con clausola `WITHHELD` il prompt vieta esplicitamente di valutare cosa il
caregiver ha detto, accettato, rifiutato o rimandato, e cosa contiene il diff rispetto al branch:
si può concludere una cosa sola, che il chatbot non ha prodotto il trigger, e questo cappa
l'obiettivo a `partial`.

### 3.2 Il clamp deterministico

`test.py:clamp_undelivered_branch` alza a `partial` qualunque obiettivo che il judge ha bocciato
mentre la clausola non era stata consegnata, ricalcola l'esito complessivo e registra cosa ha
cambiato in `branch_clamped`. Deliberatamente stretto: solo `failed → partial`, mai tocca
`completed`.

Non sostituisce la §3.1, è il pavimento sotto di essa: un modello debole nel ruolo di judge non
deve poter trasformare un limite dell'harness in un fallimento del sistema sotto test.

### 3.3 Altre due regole nel judge

- **Obiettivi soppiantati** (scenario 30): `APPLIED CHANGES` confronta inizio e fine, quindi può
  mostrare solo l'ultimo valore di un campo. "Aggiungi un check di 30 min alle 11:00" seguito da
  "portalo a 45 min alle 11:30" lascia una sola attività alle 11:30, e leggerlo come fallimento
  del primo obiettivo rende ogni sequenza di questa forma impossibile da superare.
- **Branch che prescrivono una sostituzione**: *"ask for a safer alternative, then accept
  whatever the assistant suggests"*. Lì la sostituzione **è** lo stato atteso, e l'attività
  nominata prima dell'`If…` si aspetta di *non* esistere. Vale solo per un branch `DELIVERED`.

### 3.4 Modifiche dichiarate e mai applicate

`Chat._record_unsupported_claim` più `utils.claims_applied_change` segnalano una risposta che
annuncia una modifica quando nessuna write è riuscita nel turno. Colonna `unsupported_claims`.

**Non è un gate e non è un voto**: la metà testuale non decide niente ed è accoppiata alla metà
deterministica — se nel turno una write è riuscita, non scatta. È per questo che qui un match su
frase è accettabile dove per il gate di consegna era stato scartato. Un falso positivo costa una
riga di report che un revisore scarta, non una clausola consegnata per un problema inesistente.

Il caso che l'ha motivato è lo scenario 48, dove il caregiver chiede conferma esplicita:

> **CAREGIVER:** *just to confirm: the vital-signs check now lasts 25 minutes…?*
> **CHATBOT:** *Yes — Vital signs check – 25 minutes starting at 08:20 each day.*

Il diff dice 10 minuti. Sul batch dei 23 il rilevatore ha ripescato esattamente lo scenario 48.

### 3.5 Altro nell'harness

- **`--ids`** in `test.py`: `--ids 3-8,26,42-49`, valida contro gli scenari esistenti e nomina la
  selezione effettiva nell'header del log. Prima l'unico modo di rieseguire un sottoinsieme era
  un range contiguo. Su `gpt-oss-20b` uno scenario costa 6-9 minuti, quindi rieseguirne 23
  invece di 50 non è un dettaglio.
- **Guardia sull'exit al primo messaggio**, in `test.py` e in `agent_graph.py` per tenere i due
  driver allineati. `is_exit_message` matcha una keyword *finale* — di proposito, perché il
  caregiver chiude i turni veri con "Thanks! exit" — e quella tolleranza rende fatale un primo
  messaggio che finisce con exit. Misurato su `gpt-oss-20b`, scenario 1: il caregiver ha scritto
  entrambi gli obiettivi **e** `exit` in un colpo solo, prima che l'assistente parlasse; 1 turno,
  2 richieste, `failed` su diff vuoto. `utils.strip_exit_keyword` toglie la keyword così la
  richiesta arriva pulita.
- **Prompt del caregiver**: non presentare liste di opzioni all'assistente né ripetere le sue
  (l'inversione di ruolo dello scenario 8); non scrivere come se fosse l'assistente; non
  alterare i parametri numerici del proprio obiettivo se non dopo un blocco con alternative
  offerte.
- **Warning a inizio batch** quando caregiver+judge girano sul modello sotto test. La
  configurazione resta all'operatore: `SIM_PROVIDER`/`SIM_MODEL` esistono per separarli.
- **Colonne nuove** nel report: `branch_clamped`, `safety_verdicts`, `unsupported_claims`.
  `_sync_headers` le aggiunge in coda a un workbook esistente senza spostare le righe già
  salvate.

---

## 4. Dati e scenari

- **`data/medicines/`**: aggiunte `atenolol.md`, `bisoprolol.md`, `cetirizine.md`. Lo scenario 4
  chiedeva un beta-bloccante alternativo al propranololo e in KB non ce n'era nessuno
  cardioselettivo: l'obiettivo era insoddisfacibile per costruzione. Verificato che i nuovi
  documenti sono raggiungibili — una query su "Propranolol" restituisce anche atenololo
  (`topic`, d=0.694) e bisoprololo (d=0.717).
- **`scenarios/8.json`**: l'attività si chiamava `Antihistamine`, una classe e non un farmaco,
  quindi `_identifying_tokens` non poteva matchare e `medicine_not_found` era garantito. Il
  temporal ordering che lo scenario voleva misurare non veniva mai raggiunto. Rinominata
  `Cetirizine (antihistamine)`.
- **`scenarios/26.json`**: la clausola diceva *"cambia in modo che avvenga dopo colazione"* e il
  caregiver l'ha letta come "rimettilo a 08:20" — un no-op. Ora nomina il cambio di dipendenza e
  mantiene l'orario richiesto.
- **`scenarios/6.json`**: la clausola finiva con il caregiver che fa una domanda e non nominava
  alcuno stato atteso, quindi qualunque esito era difendibile e il judge bocciava la sostituzione
  che la clausola stessa aveva invitato. Aggiunto il *"then follow the assistant's recommendation
  and confirm the final activity"* che gli altri scenari della famiglia hanno già.

---

## 5. Backend OpenRouter

`config_loader.py` e `llm_client.py` non conoscevano il provider: `PROVIDER=openrouter` sarebbe
fallito in validazione. Aggiunti costanti, provider supportato, credenziali, modello di default e
catena di inferenza. In `llm_client.py` c'era un solo punto provider-specifico da toccare — il
controllo della chiave — perché tutto il resto parla già protocollo OpenAI.

Limiti di default a 0: su OpenRouter il limite è una proprietà dell'account e scala col credito,
non c'è una cifra da hardcodare. Eccezione i listing `:free`, capati a 20 req/min lato server.

Tre problemi trovati eseguendo:

- **HTTP 402 moriva con un traceback.** Ora è mappato su `DailyQuotaExceeded`, che `test.py` già
  intercetta per fermare il batch conservando gli scenari valutati: è un saldo, non un rate
  limit, e nessuna attesa lo risolve.
- **I nomi delle tool call arrivano sporchi.** OpenRouter che serve gpt-oss incolla il marcatore
  del formato *harmony* al nome della funzione: nello scenario 1, **11 tool call su 22** arrivavano
  come `add_therapy_activity<|channel|>commentary`, intermittentemente, nello stesso run di altre
  pulite — quindi metà delle azioni finiva in `Tool not found`. Gli argomenti erano JSON valido.
  `utils.clean_tool_name` taglia dal primo carattere che non può stare in un identificatore, e
  ogni correzione lascia un warning. Non inventa tool: nello stesso run `get_medicine_activities`,
  che il modello si era inventato, è stato correttamente rifiutato.
- **`openai/gpt-oss-20b:free` non esiste più** (404, *"The paid version is available now"*). Il
  listing a pagamento costa 0.03$/M prompt e 0.13$/M completion: ~0.005$ per scenario, ~0.07$ per
  un batch di 23.

### 5.1 Il seed non fa niente su questi backend — misurato

Stesso prompt, `seed=42`, ripetuto:

| backend | risultato |
|---|---|
| `openrouter/openai/gpt-oss-20b`, routing default | **5 output distinti su 5**, serviti da 4 upstream diversi (CoreWeave, SiliconFlow, Amazon Bedrock, Novita) |
| stesso, upstream fissato (`provider.order`, no fallback) | **3 su 3 distinti** su Novita, su SiliconFlow e su CoreWeave separatamente |
| `openai/gpt-5.4-mini` | **3 su 3 distinti**, e nessun `system_fingerprint` in risposta |

OpenRouter dichiara `seed` fra i `supported_parameters` di quel modello: viene accettato e
ignorato. Il batching continuo su uno stack condiviso rende la ripetibilità bit-per-bit
indisponibile a prescindere.

Questo **contraddice un'assunzione di `NOTE-BRANCH-fix-after-tests.md`**, dove il judge con
`SIM_SEED` fisso è raccomandato come *"senza contropartite"* perché renderebbe confrontabile una
rivalutazione. Regge solo se il seed funziona. La misura è registrata accanto alla manopola — nel
docstring di `LLMConfig` e accanto a `SEED` in `.env.example` — perché è il tipo di assunzione che
qualcuno rifarà.

**Conseguenza**: l'unico modo di attribuire un cambiamento al codice invece che al campionamento
resta N ripetizioni con la dispersione riportata, per il judge come per il therapy manager.

### 5.2 Perché è lento (misurato)

Non è throttling. Scenario 4: 40 richieste, 480s, mediana 9s per richiesta, 164K token. Il 78%
delle richieste è il sistema sotto test (loop del manager più delegazioni al checker), il 22%
l'harness (un caregiver per turno più il judge).

Il moltiplicatore è `reasoning_effort`. Su una chiamata banale (prompt di 71 token, "say ok"):
**7,6s a `high` contro 1,1s a `low`**, con 53 token di reasoning contro 12. Sui prompt reali —
mediana 3953 token, perché system prompt e schemi dei tool vengono rimandati a ogni iterazione —
il ragionamento si allunga di conseguenza.

---

## 6. Verifica eseguita

**Offline, senza LLM**: 23 asserzioni sul gate (check mancante, verdetto su un altro farmaco,
remark che non blocca, blocking permanente e suo rilascio per verdetto più recente, caution che
segnala e non blocca, update/remove, edit di sola descrizione, fail-open su formato,
`safety_check_required` che non apre il gate). Più i rilevatori testuali su repliche reali prese
dai log: 14/14 su `claims_applied_change`, incluso il falso positivo *"nothing was created"*
trovato in un run live e corretto con la gestione della negazione.

**Live su `openai/gpt-5.4-mini`**, scenari 3–8 e 32. **Live su `openrouter/openai/gpt-oss-20b`**
(stessi pesi del batch originale), scenario 1 e i 23 scenari problematici a effort `low`.

Sul batch dei 23:

| | |
|---|---|
| migliorati | 9 — 6, 8, 15, 18, 32, 34, 36, 46, 47 |
| peggiorati | 4 — 10, 30, 43, 44 |
| uguali | 10 — 3, 4, 13, 14, 26, 37, 42, 45, 48, 49 |

**Questi numeri non sono attribuibili alle modifiche.** Il seed non funziona (§5.1) e l'effort è
passato da `high` a `low` fra i due run: le cause si sovrappongono e N=1 non le separa. La prova
di quanto sia labile è la riesecuzione dei cinque scenari dopo la rimozione del latch — 3 e 13
passati a `completed`, 32 e 34 passati a `failed`, con la nota del judge sul 32 che dice *"the
caregiver ultimately chose to consult a clinician"*, cioè puro campionamento.

Quello che invece **è** solido, perché è meccanismo e non esito:

- **21/23 clausole condizionali consegnate**, contro 6/33 nel batch originale — e quelle 6 su una
  causa sbagliata. Sui trigger soft era 0/18.
- **Il check di sicurezza non è più saltabile**: `safety_checks_skipped` ha contato 2 tentativi di
  scrittura prima del controllo (scenari 10 e 32), entrambi rifiutati dal codice. Nel batch
  originale gli stessi tentativi diventavano farmaci controindicati scritti in terapia.
- **`unsupported_claims` ha ripescato lo scenario 48**, il caso della conferma falsa.
- **`safety_verdicts_unparsed`**: 4 scenari su 23 a `low` (§1.3 bis).

---

## 7. Aperto

- **Scenario 4** non risolto: in un run su gpt-5.4-mini ha aggiunto Atenololo correttamente, in
  un altro ha dichiarato di non poter verificare alcuna alternativa. Il buco di dati è chiuso, ma
  il checker propone l'alternativa dai documenti già recuperati in modo incostante. Servirebbe un
  modo di *elencare* i farmaci in KB: è una capability nuova, non aggiunta.
- **Scenario 7** era `completed` e ora è `partial`: il checker annuncia "Metformin not in current
  therapy" al turno 1, quindi il manager non passa mai la dipendenza inesistente e
  `missing_dependency` non scatta più. `CLAUDE.md` documentava già quella guardia come
  praticamente irraggiungibile; ora lo è per costruzione. Da decidere se riscrivere 7, 57 e 89 o
  accettare che misurino solo la condotta.
- **Rimozioni non richieste**: i run hanno prodotto due attività rimosse che nessuno aveva chiesto
  (una Cetirizina, un "Afternoon walk"), catturate da `changed_activities`. Difetto reale del
  modello, ora visibile.
- **Scenario 10**: `safety_checks_skipped=2` e **due Ibuprofen duplicati** nel diff. Dopo il
  rifiuto il modello ha ritentato creando una seconda attività invece di riprovare la stessa.
- **Ripetizioni**: nulla di quanto sopra diventa attribuibile senza N run per configurazione. A
  0,07$ per 23 scenari, 5 ripetizioni costano ~0,35$.

---

## 8. Inventario file

**Nuovi**

| file | |
|---|---|
| `src/safety.py` | severità tipizzata, parsing del verdetto, identità dell'attività (260 righe) |
| `data/medicines/atenolol.md` · `bisoprolol.md` · `cetirizine.md` | monografie mancanti |

**Modificati**

| file | cosa |
|---|---|
| `src/chat.py` | gate di sicurezza, registrazione verdetti, claim non supportati, pulizia nomi tool |
| `src/test.py` | consegna clausola al judge, clamp, `--ids`, guardia exit, colonne nuove, warning stesso-modello |
| `src/utils.py` | `claims_applied_change`, `clean_tool_name`, `strip_exit_keyword` |
| `src/agents/check_agent.py` | verdetto tipizzato, regole su severità e alternative |
| `src/agents/therapy_manager_agent.py` | la decisione è del caregiver; alternative farmacologiche |
| `src/agents/judge_agent.py` | stato di consegna della clausola, obiettivi soppiantati, branch di sostituzione |
| `src/agents/caregiver_agent.py` | non fare l'assistente; non alterare i propri parametri |
| `src/config_loader.py` | provider `openrouter`; misura sul seed |
| `src/llm_client.py` | chiave openrouter; HTTP 402 → `DailyQuotaExceeded` |
| `src/agent_graph.py` | guardia exit, allineata a `test.py` |
| `src/results_extractor.py` | `branch_clamped`, `safety_verdicts`, `unsupported_claims` |
| `scenarios/6.json` · `8.json` · `26.json` | clausole e nomi (una riga ciascuno) |
| `CLAUDE.md` · `.env.example` | architettura e configurazione aggiornate |

`old_results.xlsx` risulta cancellato: era già così a inizio sessione. `.env.example` contiene
anche un blocco OpenRouter aggiunto dall'operatore. `all_results_test_50.xlsx` (il batch di
partenza) è untracked.

# Colonne del foglio Excel dei risultati

File: `logs/batch_results/all_results.xlsx`, foglio **Results** — una riga per scenario.
Le colonne sono definite in `src/results_extractor.py` (`_OBJECTIVE_COLUMNS`) e popolate in
`src/test.py` (`run_scenario`) a partire dal verdetto del `JudgeAgent` arricchito dall'harness.

Il file è **cumulativo**: ogni batch appende righe senza toccare quelle precedenti. Una
colonna aggiunta da una versione più recente del codice viene accodata in fondo al foglio
esistente (`_sync_headers`), quindi le righe sono sempre scritte per nome di colonna e mai
per posizione. Le celle sono troncate a 32767 caratteri (limite del formato xlsx) con una
nota `[… truncated]` in coda.

---

## Identificazione della run

| Colonna | Significato |
|---|---|
| `test_date` | Data/ora in cui la riga è stata scritta |
| `batch_id` | Identificativo del batch (es. `20260630_143000`) |
| `scenario_id` | Numero dello scenario in `scenarios/` |
| `patient` | `Nome Cognome(patient_id)` del paziente dello scenario |

## Esito e costi

| Colonna | Significato |
|---|---|
| `overall_status` | Verdetto complessivo del `JudgeAgent`: `completed` / `partial` / `failed` / `not_attempted` / `error`. Colora l'intera riga (verde / giallo / rosso / grigio / rosso scuro) |
| `turns` | Numero di turni di conversazione consumati |
| `elapsed_seconds` | Durata dello scenario in secondi |

## Blocco deterministico — ciò che il codice constata, non ciò che un modello giudica

È la parte che un revisore dovrebbe poter leggere senza aprire i log: sono tutti dati
calcolati in codice durante la run.

| Colonna | Significato |
|---|---|
| `changed_activities` | Elenco nominale di **tutte** le attività toccate dalla conversazione. È la colonna da scorrere per beccare una modifica che nessuno aveva chiesto |
| `issue_signals` | Le cause bloccanti sollevate dal sistema stesso: `schedule_conflict`, `missing_dependency`, `temporal_ordering`, `dependency_blocked`, medicina non trovata, più i rifiuti del safety gate `safety_blocked` / `safety_caution` / `safety_check_required`. `none` se nulla ha bloccato |
| `branch_outcome` | Se il ramo condizionale dello scenario è stato attivato: `exercised` (l'assistente ha sollevato il punto da solo e la clausola condizionale gli è stata quindi consegnata), `not_raised_but_change_applied` (non l'ha sollevato ma ha comunque modificato — verosimilmente ha aggirato il problema), `not_raised_no_change`, oppure `n/a` se lo scenario non ha clausola condizionale |
| `branch_clamped` | `no`, oppure `objectives [n] failed→partial`: il judge ha bocciato un obiettivo la cui clausola condizionale non era mai stata consegnata al caregiver, e l'harness ha alzato il voto a `partial`. Segnala un limite dell'harness, non una colpa del sistema testato (`test.clamp_undelivered_branch`) |
| `safety_verdicts` | Ogni verdetto del checker con turno, severità (`blocking` / `caution` / `remark`, con `(untyped)` se non tipizzato) e nome attività. È qui che si legge **perché** una scrittura è stata rifiutata — o perché una che andava rifiutata non lo è stata. Eventuale riga finale `[!]`: verdetti non parsati (il gate ha fallito in apertura) o scritture tentate prima di qualsiasi controllo |
| `unsupported_claims` | Risposte in cui l'assistente ha annunciato una modifica che nessuna scrittura ha effettivamente eseguito, con il numero di turno. `none` se nessuna |
| `history_warnings_retrieved` | I rischi che il RAG ha messo davanti all'assistente (solo livello warning), da leggere contro il transcript per capire se sono stati riferiti al caregiver |

## Obiettivi

| Colonna | Significato |
|---|---|
| `objectives_scripted` | Quanti obiettivi lo script dello scenario prevedeva |
| `objectives_status` | Stringa compatta con l'iniziale dello stato di ogni obiettivo giudicato, es. `C,P,F` (Completed / Partial / Failed / Not attempted). Confrontata con `objectives_scripted` distingue "fallito" da "mai chiesto dal caregiver" |
| `objectives` | Il testo dello scenario / script degli obiettivi (l'input) |
| `judge_check` | JSON completo della valutazione per obiettivo del judge (stato + note). Se `overall_status = error` contiene invece il messaggio d'errore e i primi 500 caratteri dell'output grezzo non parsato del judge |

## Materiale grezzo per l'ispezione

| Colonna | Significato |
|---|---|
| `applied_changes` | Il diff programmatico completo iniziale→finale, cioè esattamente ciò che è stato mostrato al judge (la valutazione è diff-based, non basata sul transcript) |
| `conversation` | Il transcript completo della conversazione |
| `initial_therapy` | JSON della terapia installata prima della conversazione |
| `final_therapy` | JSON della terapia a fine conversazione (vuoto in caso di `error`) |

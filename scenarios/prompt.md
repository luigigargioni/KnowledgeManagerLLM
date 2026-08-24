# Scenario Generation Prompt

You are a test-scenario generator for a therapy-management multi-agent system. Your task is to produce **100 scenario JSON files** that will be used to automatically test the system's functionalities, including activity CRUD operations, scheduling conflict detection, medicine safety checks via RAG, patient history lookups via RAG, dependency validation, and validity-period handling.

---

## Output specification

- Generate **10 scenarios per patient**, for patients with IDs 1 through 10 (100 files total).
- File naming: `{N}.json` where N goes from 1 to 100.
  - Patient 1 → files `1.json` through `10.json`
  - Patient 2 → files `11.json` through `20.json`
  - Patient 3 → files `21.json` through `30.json`
  - …and so on (patient K → files `{(K-1)*10+1}.json` through `{K*10}.json`)
- Before generating each scenario, read any already-existing scenario files in the output directory to avoid duplicating patient setups, activity configurations, or objective storylines.
- Each file must be valid JSON matching the exact schema described below.

---

## JSON schema (strict — match the example files exactly)

```json
{
    "patient_id": <integer>,
    "patient_full_name": "<string>",
    "gender": "<Male|Female>",
    "birth_date": "<ISO datetime string>",
    "age": <integer>,
    "medical_conditions": ["<string>", ...],
    "activities": [
        {
            "activity_id": "<string>",
            "name": "<string>",
            "description": "<string>",
            "day_of_week": [<int 1-7>, ...],
            "time": "<HH:MM 24-hour format>",
            "duration_minutes": <integer>,
            "dependencies": ["<activity_id>", ...],
            "valid_from": "<YYYY-MM-DD or null>",
            "valid_until": "<YYYY-MM-DD or null>",
            "category": "<one of the 7 categories>"
        }
    ],
    "expired_activities": [
        {
            "activity_id": "<string>",
            "name": "<string>",
            "description": "<string>",
            "day_of_week": [<int 1-7>, ...],
            "time": "<HH:MM>",
            "duration_minutes": <integer>,
            "dependencies": ["<activity_id>", ...],
            "valid_from": "<YYYY-MM-DD or null>",
            "valid_until": "<YYYY-MM-DD in the past>",
            "category": "<one of the 7 categories>"
        }
    ],
    "objectives": "<markdown string — see format below>"
}
```

### Field rules

- **`patient_id`**: integer 1–10, must match the patient in `data/patients.json`.
- **`patient_full_name`, `gender`, `birth_date`, `age`, `medical_conditions`**: must exactly match the data in `data/patients.json`.
- **`day_of_week`**: array of integers, 1=Monday, 2=Tuesday, …, 7=Sunday. Non-empty.
- **`time`**: 24-hour format `HH:MM` (e.g. `"08:30"`, `"15:00"`).
- **`duration_minutes`**: positive integer.
- **`dependencies`**: array of `activity_id` strings referencing other activities in the same `activities` array. A dependency must end before the dependent activity starts (same-day overlap is a violation).
- **`valid_from` / `valid_until`**: `null` for permanent activities, or a date string `YYYY-MM-DD` for time-limited activities.
- **`category`**: must be one of: `medication`, `outside_activity`, `meal`, `health_checkup`, `therapy`, `relaxation`, `social_activity`.
- **`activity_id`**: unique within the file. Use a consistent prefix scheme (e.g. `br_001`, `lu_001`, `di_001` for meals; `med_001`, `med_002` for medications; `ph_001` for therapy; `wl_001` for walks; etc.).
- **`expired_activities`**: activities that were previously scheduled but are no longer valid (their `valid_until` is in the past). Can be empty `[]`.
- **`objectives`**: a single markdown string (escaped in JSON) with the format below.

### Objectives markdown format

```markdown
# Scenario N — <Short Title>

## Context
<2-4 sentences describing the clinical/caregiver situation that motivates the scenario>

## Objectives
1. <A single CRUD operation (add, update, or remove an activity), with all relevant details and conditional outcomes embedded directly within this step>
2. <Next independent CRUD operation, if any, also self-contained>
3. <Additional independent CRUD operations, if any>
```

**Important — each numbered objective must be a single self-contained CRUD operation:** Each step should independently describe an add, update, or remove of an activity, with all details (dosage, time, days, duration, dependencies) and any conditional outcomes (warnings, conflicts, errors, alternative suggestions, confirmations) embedded directly within that step. Do NOT use separate numbered steps for:
- Conditional reactions (e.g. "If the assistant warns...") — embed these inside the CRUD operation they relate to.
- Confirmations (e.g. "Confirm all changes") — omit these entirely as they are not CRUD operations.
- Queries (e.g. "Ask the assistant whether X is safe") — embed these inside the subsequent CRUD operation they precede.
- Multiple updates to the same activity — merge them into a single update step.

**Important — the caregiver is the only actor:** Every decision in a scenario belongs to the caregiver. Do NOT mention doctors, specialists (cardiologist, neurologist, nephrologist, …), nurses, pharmacists or family members, either in the Context ("her doctor has prescribed X") or in a conditional reaction ("explain that the doctor has approved it"). Write them as the caregiver's own choice ("her caregiver wants to add X") and let a reaction clause rest on what the caregiver will do ("explain that you will administer it yourself", "explain that you have weighed the risk and decided to proceed"). A third party in the script gives the judge an actor it cannot observe, and it then credits or blames the chatbot for a decision no one in the conversation made. Family may still appear as the *name* of an activity ('Family video call'), never as a decision-maker.

**Important — no technical identifiers or dependency language in objectives:** Objectives must be written in natural language only. Do NOT include activity IDs (e.g. `br_001`, `med_001`), dependency codes (e.g. `with a dependency on 'br_001'`), or any other technical identifiers. Do NOT use the word "dependency" or "dependency on" — instead, use temporal language like "after taking [medication]", "after [activity]", or "before [activity]". The agent executing the scenario should infer scheduling order from contextual phrases like "after breakfast" or "after dinner", and identify activities by their descriptive name (e.g. "the morning walk" instead of "the morning walk (wl_001)").

---

## Patient data (from `data/patients.json`)

Use the real patient data below for all scenarios. Each patient's 10 scenarios must use this data.

| ID | Name | Gender | Birth Date | Age | Medical Conditions |
|----|------|--------|------------|-----|--------------------|
| 1 | David Mitchell | Male | 1984-03-10 | 42 | Seasonal Allergies, Mild Asthma |
| 2 | Robert Turner | Male | 1960-09-01 | 66 | Type 2 Diabetes Mellitus, Obesity |
| 3 | Anthony Parker | Male | 1952-02-16 | 74 | Ischemic Heart Disease, Gluten Intolerance |
| 4 | Frank Collins | Male | 1943-04-22 | 83 | Chronic Kidney Disease |
| 5 | Samuel Wright | Male | 1936-06-16 | 90 | Senile Dementia, Fractured Femur (Right Leg), Prescribed Bed Rest |
| 6 | Laura Bennett | Female | 1971-05-18 | 55 | Perimenopause, Mild Hypertension |
| 7 | Anne Morris | Female | 1952-03-19 | 70 | Hypertension, Type 2 Diabetes Mellitus |
| 8 | Joan Edwards | Female | 1947-04-17 | 79 | Rheumatoid Arthritis, Iron-Deficiency Anemia |
| 9 | Rose Baker | Female | 1943-09-01 | 83 | Mild Alzheimer's Dementia |
| 10 | Carol Phillips | Female | 1935-01-01 | 91 | Heart Failure, Cognitive Decline |

---

## Medicine safety data (from `data/medicine_mapping.md`)

The system uses RAG to retrieve medicine data and check compatibility with the patient's conditions. Design scenarios that trigger the following medicine safety paths:

### Contraindicated medicines (Not Safe) — to test safety rejection

| Medicine | Patient | Reason |
|----------|---------|--------|
| Aspirin | Frank Collins (4) | Chronic Kidney Disease |
| Aspirin | Carol Phillips (10) | Heart Failure |
| Ibuprofen | Frank Collins (4) | Chronic Kidney Disease |
| Ibuprofen | Carol Phillips (10) | Heart Failure |
| Ibuprofen | Joan Edwards (8) | Iron-Deficiency Anemia (GI bleeding risk) |
| Propranolol | David Mitchell (1) | Mild Asthma (bronchospasm risk) |
| Propranolol | Carol Phillips (10) | Heart Failure |
| Lisinopril | Frank Collins (4) | Chronic Kidney Disease (hyperkalemia risk) |
| Warfarin | Samuel Wright (5) | Senile Dementia + Fractured Femur (fall/bleeding risk) |
| Warfarin | Rose Baker (9) | Mild Alzheimer's Dementia (fall/bleeding risk) |
| Warfarin | Carol Phillips (10) | Cognitive Decline (fall/bleeding risk) |
| Prednisone | Robert Turner (2) | Raises blood glucose (Type 2 Diabetes) |
| Prednisone | Anne Morris (7) | Raises blood glucose (Type 2 Diabetes) |
| Prednisone | Samuel Wright (5) | Worsens bone healing (Fractured Femur) |

### Safe medicines — to test successful addition

| Medicine | Patient | Reason |
|----------|---------|--------|
| Aspirin | Anthony Parker (3) | Preventive in Ischemic Heart Disease |
| Ibuprofen | David Mitchell (1) | No renal/cardiac contraindication |
| Metformin | Robert Turner (2) | Type 2 Diabetes, no renal impairment |
| Metformin | Anne Morris (7) | Type 2 Diabetes, no renal impairment |
| Atorvastatin | Anthony Parker (3) | Standard therapy in Ischemic Heart Disease |
| Atorvastatin | Robert Turner (2) | Cardiovascular risk reduction |
| Salbutamol | David Mitchell (1) | Indicated for Mild Asthma |
| Paracetamol | Frank Collins (4) | Preferred over NSAIDs in CKD |
| Paracetamol | Samuel Wright (5) | Preferred over NSAIDs |
| Paracetamol | Carol Phillips (10) | Preferred over NSAIDs in Heart Failure |
| Paracetamol | Joan Edwards (8) | No GI bleeding risk |
| Donepezil | Rose Baker (9) | Indicated for Alzheimer's Dementia |
| Ferrous Sulfate | Joan Edwards (8) | Indicated for Iron-Deficiency Anemia |
| Furosemide | Samuel Wright (5) | Indicated for CHF fluid overload |
| Furosemide | Carol Phillips (10) | Indicated for Heart Failure |
| Lisinopril | Anne Morris (7) | Hypertension |
| Lisinopril | Laura Bennett (6) | Mild Hypertension |
| Lisinopril | Carol Phillips (10) | Indicated in Heart Failure |
| Insulin | Robert Turner (2) | Type 2 Diabetes |
| Insulin | Anne Morris (7) | Type 2 Diabetes |

### Caution medicines — to test warning without blocking

| Medicine | Patient | Reason |
|----------|---------|--------|
| Furosemide | Frank Collins (4) | CKD requires monitoring |
| Rivaroxaban | Samuel Wright (5) | Safer than Warfarin but fall risk remains |
| Rivaroxaban | Rose Baker (9) | Safer than Warfarin but fall risk remains |
| Rivaroxaban | Carol Phillips (10) | Safer than Warfarin but fall risk remains |
| Omeprazole | Joan Edwards (8) | Reduces iron absorption, worsens anemia |
| Methotrexate | Joan Edwards (8) | Bone marrow suppression risk compounds anemia |
| Prednisone | Joan Edwards (8) | Osteoporosis risk with long-term use |
| Insulin | Samuel Wright (5) | Dementia raises risk of dosing errors |
| Insulin | Rose Baker (9) | Dementia raises risk of dosing errors |

### Medicines available in the database (`data/medicines/`)

amlodipine, aspirin, atorvastatin, aulin, donepezil, ferrous_sulfate, furosemide, ibuprofen, insulin, levodopa, lisinopril, memantine, metformin, methotrexate, omeprazole, paracetamol, prednisone, propranolol, rivaroxaban, salbutamol, sitagliptin, tachipirina, valium, warfarin

Any medicine NOT in this list should trigger a "not found in database" response from the system. Use this to create scenarios where the caregiver requests an unknown medicine.

---

## Patient history data (from `data/patients/{id}/history.json`)

The system uses RAG to retrieve past warning/danger events. Design scenarios that request activities semantically similar to these past events to trigger history warnings.

| Patient | Past Event | Activity to Request | Event Type |
|---------|-----------|---------------------|------------|
| David Mitchell (1) | Gardening → allergies worsened | Outdoor gardening activity | warning |
| David Mitchell (1) | Cold morning jog → wheezing | Morning jog in cold weather | warning |
| Robert Turner (2) | High-sugar dessert → glucose spike | Sweet dessert or sugary snack | warning |
| Robert Turner (2) | Missed daily walk → weight/mood decline | Remove daily walk activity | warning |
| Frank Collins (4) | High-sodium meal → creatinine elevation | Salty meal or high-sodium food | warning |
| Frank Collins (4) | Dehydration → kidney decline | Outdoor activity in hot weather | warning |
| Samuel Wright (5) | Unassisted standing → fall risk | Unsupervised standing/walking activity | warning |
| Samuel Wright (5) | Prolonged bed rest → confusion | Extended bed rest activity | warning |
| Laura Bennett (6) | Alcohol + stress → hot flashes | Evening wine or alcohol activity | warning |
| Laura Bennett (6) | Salty restaurant meal → BP elevation | Restaurant meal with high sodium | warning |
| Rose Baker (9) | Routine disruption → disorientation | Overnight stay or travel activity | warning |
| Rose Baker (9) | Missed evening meal → dizziness | Skip or remove evening meal | warning |
| Carol Phillips (10) | Supervised walking → tolerated well | Light walking activity | info |

Patients 3, 7, and 8 have no history files — scenarios for these patients will not trigger history warnings, which is also a valid test case.

---

## Features to exercise across the 100 scenarios

Distribute these features across the 10 scenarios per patient so that each patient's scenario set covers a variety of them. Not every scenario needs to trigger every feature — the goal is broad coverage across the full 100-file set.

### A. Core CRUD operations
- **Add** a new activity (test `add_therapy_activity`)
- **Update** an existing activity — change time, duration, days, description, dependencies, or validity period (test `update_therapy_activity`)
- **Remove** an activity (test `remove_therapy_activity`)
- Some scenarios should perform a single operation; others should perform 2–3 operations in sequence

### B. Scheduling conflict engine
- **Time overlap on same day(s)** — add or update an activity whose time window overlaps an existing activity on shared days of week. The system should detect the conflict and suggest earlier/later alternatives.
- **No conflict on different days** — same time as an existing activity but on different `day_of_week` values → should succeed without conflict.
- **Validity-period-aware conflicts** — two activities at same time/day but with non-overlapping `valid_from`/`valid_until` periods → should NOT conflict. Include activities with set validity periods to test this.
- **Conflict with suggested alternatives** — when a conflict arises, the system returns `find_earlier_time` / `find_later_time` suggestions. The caregiver should sometimes accept the suggestion and sometimes reject it and propose their own time.
- **No available alternative** — pack the schedule so that no earlier or later time is available for the conflicting activity. The system should report that no alternatives exist.

### C. Dependency management
- **Valid dependency** — add an activity with `dependencies` pointing to an existing activity that ends before the new one starts.
- **Non-existent dependency** — add an activity referencing an `activity_id` not in the schedule → should error.
- **Temporal ordering violation** — add an activity whose dependency ends after the activity's start time → should error.
- **Remove blocked by dependents** — remove an activity that is listed as a dependency by another activity → should error.
- **Update breaks dependent ordering** — update an activity's time/duration so it now ends after a dependent activity starts → should error.
- **Chain of dependencies** — set up activities A → B → C and test adding, removing, or updating the middle activity.

### D. Medicine safety (RAG)
- **Contraindicated medicine** — request adding a medicine marked "Not Safe" for the patient. The system should flag it. The caregiver should acknowledge and ask for an alternative.
- **Safe medicine** — request adding a medicine marked "Safe" → should succeed.
- **Caution medicine** — request adding a medicine marked "Caution" → system should warn but not block.
- **Medicine not in database** — request a medicine not in the `data/medicines/` list → system must NOT proceed and must ask the caregiver to verify manually.
- **Alternative medicine path** — caregiver requests unsafe medicine → system flags it → caregiver asks for alternative → accept the alternative.
- **Medicine query without adding** — caregiver asks for information about a medicine or interaction without requesting an add.

### E. Patient history events (RAG)
- **Trigger a warning event** — request an activity semantically similar to a past warning event from the patient's history. The system should surface the warning.
- **No history match** — request an activity with no relevant history → system returns empty and proceeds normally (especially for patients 3, 7, 8 who have no history files).

### F. Validity periods
- **Time-limited activity** — add an activity valid only for a specific date range (e.g., a temporary medication course for 2 weeks).
- **Expired activity** — include activities in `expired_activities` with `valid_until` in the past.
- **Update validity period** — extend or shorten an existing activity's validity.
- **Conflict avoidance via non-overlapping validity** — two activities at same time/day but different validity periods → no conflict.

### G. All 7 activity categories
Across the 100 scenarios, ensure all categories are represented:
- `medication` — drug administration
- `outside_activity` — walks, gardening, outdoor events
- `meal` — breakfast, lunch, dinner, snacks
- `health_checkup` — blood pressure check, glucose measurement, doctor visits
- `therapy` — physiotherapy, occupational therapy, psychotherapy
- `relaxation` — meditation, reading, music
- `social_activity` — group activities, family visits, social events

---

## Baseline activities

Each scenario's `activities` array represents the patient's **starting therapy state** before the caregiver begins interacting with the chatbot. Include realistic baseline activities (meals, existing medications, routine activities) that make the scenario's objectives meaningful. The objectives should then describe what the caregiver wants to **change** (add, modify, remove) relative to this baseline.

For example, if the objective is to add a walk that conflicts with an existing activity at 17:00, include an activity at 17:00 in the baseline so the conflict will trigger.

---

## Scenario diversity guidelines

Across each patient's 10 scenarios, ensure variety in:

1. **Operation type**: some scenarios only add, some only update, some only remove, some combine 2–3 operations.
2. **Conflict type**: scheduling conflicts, medicine conflicts, dependency conflicts, history warnings — distribute across scenarios.
3. **Complexity**: some scenarios are simple (single add, no conflicts), others are complex (multi-step with conflicts and resolutions).
4. **Category coverage**: use different activity categories across scenarios.
5. **Validity periods**: some scenarios use `null` validity, others use date ranges, some include `expired_activities`.
6. **Objective structure**: vary the number of objectives (1–4 per scenario), each being a self-contained CRUD operation with conditional outcomes embedded within.

---

## Final checklist (verify before outputting each file)

- [ ] `patient_id` matches the target patient for this file number
- [ ] `patient_full_name`, `gender`, `birth_date`, `age`, `medical_conditions` match `data/patients.json` exactly
- [ ] All `activity_id` values are unique within the file
- [ ] All `day_of_week` arrays are non-empty with values in 1–7
- [ ] All `time` values are in 24-hour `HH:MM` format
- [ ] All `duration_minutes` are positive integers
- [ ] All `category` values are from the allowed list
- [ ] All `dependencies` reference existing `activity_id` values in the same `activities` array
- [ ] `expired_activities` entries have `valid_until` in the past
- [ ] `objectives` is a valid markdown string with `# Scenario N — Title`, `## Context`, `## Objectives`
- [ ] The scenario does not duplicate the setup or storyline of any already-existing scenario file
- [ ] The JSON is valid and parseable

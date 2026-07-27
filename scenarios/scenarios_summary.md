# Scenario Summary

## Patient 1 — David Mitchell (Seasonal Allergies, Mild Asthma)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `1.json` | Add Evening Walk and Safe Ibuprofen | Add attività + medicinale sicuro (Ibuprofen) |
| `2.json` | Update Breakfast Time with Medication Conflict | Update + conflitto con dipendenza |
| `3.json` | Add Gardening Activity with History Warning | History warning (allergies) |
| `4.json` | Add Contraindicated Propranolol | Medicinale controindicato (asma) |
| `5.json` | Add Safe Salbutamol | Medicinale sicuro |
| `6.json` | Add Cold Morning Jog with History Warning | History warning (wheezing) |
| `7.json` | Add Activity with Non-Existent Dependency | Dipendenza inesistente |
| `8.json` | Add Activity with Temporal Ordering Violation | Violazione ordinamento temporale |
| `9.json` | Add Activity with Non-Overlapping Validity Period | Evitamento conflitto via validità |
| `10.json` | Complex Multi-Step with Conflict Resolution | Add + update + conflitto + social |

## Patient 2 — Robert Turner (Type 2 Diabetes Mellitus, Obesity)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `11.json` | Add Safe Metformin | Medicinale sicuro |
| `12.json` | Update Walk, Add Atorvastatin, and Medicine Query | Update + medicinale sicuro (Atorvastatin) + medicine query |
| `13.json` | Add Sweet Dessert with History Warning | History warning (glucose spike) |
| `14.json` | Add Contraindicated Prednisone | Medicinale controindicato (diabete) |
| `15.json` | Remove Daily Walk with History Warning | History warning (missed walk) + remove |
| `16.json` | Add Insulin with Valid Dependency | Medicinale sicuro + dipendenza valida |
| `17.json` | Add Activity with Scheduling Conflict | Conflitto scheduling + accept alternative |
| `18.json` | Remove Blocked Dependency Chain Middle | Remove bloccato da dipendente |
| `19.json` | Add Time-Limited Therapy Activity | Validity period |
| `20.json` | Add Unknown Medicine Then Safe Alternative | Medicinale non in DB + safe add |

## Patient 3 — Anthony Parker (Ischemic Heart Disease, Gluten Intolerance)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `21.json` | Add Safe Aspirin | Medicinale sicuro |
| `22.json` | Add Safe Atorvastatin | Medicinale sicuro |
| `23.json` | Add Gluten-Free Snack Activity | No history match (paziente senza history warning) |
| `24.json` | Scheduling Conflict - Accept Alternative | Conflitto + accept |
| `25.json` | Scheduling Conflict - Reject Alternative | Conflitto + reject + proposta propria |
| `26.json` | Update Breaking Dependency Ordering | Violazione ordinamento dipendenza |
| `27.json` | Add Activity with Valid Dependency Chain | Catena dipendenze valida |
| `28.json` | Remove Activity Blocked by Dependent | Remove bloccato da dipendente |
| `29.json` | Expired Activity Replacement | Expired activity + validity period |
| `30.json` | Complex Multi-Step with Update and Add | Add + update + add combinati |

## Patient 4 — Frank Collins (Chronic Kidney Disease)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `31.json` | Add Safe Paracetamol | Medicinale sicuro |
| `32.json` | Add Contraindicated Aspirin | Medicinale controindicato (CKD) |
| `33.json` | Add High-Sodium Meal with History Warning | History warning (creatinine) |
| `34.json` | Add Outdoor Activity in Hot Weather with History Warning | History warning (dehydration) |
| `35.json` | Add Contraindicated Ibuprofen | Medicinale controindicato (CKD) |
| `36.json` | Add Caution Furosemide | Medicinale caution (CKD monitoring) |
| `37.json` | Add Contraindicated Lisinopril | Medicinale controindicato (hyperkalemia) |
| `38.json` | Scheduling Conflict with No Available Alternative | Schedule pieno, nessuna alternativa |
| `39.json` | Update Validity Period | Update validity period |
| `40.json` | Complex Remove, Add, and Update | Remove + add + update combinati |

## Patient 5 — Samuel Wright (Senile Dementia, Fractured Femur, Bed Rest)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `41.json` | Add Safe Paracetamol | Medicinale sicuro |
| `42.json` | Add Contraindicated Warfarin | Medicinale controindicato (dementia + fracture fall/bleeding risk) |
| `43.json` | Add Unsupervised Walking with History Warning and Safe Furosemide | History warning (fall risk) + medicinale sicuro (Furosemide) |
| `44.json` | Add Extended Bed Rest with History Warning | History warning (confusion) |
| `45.json` | Add Contraindicated Prednisone | Medicinale controindicato (worsens bone healing) |
| `46.json` | Add Caution Rivaroxaban | Medicinale caution (fall risk remains) |
| `47.json` | Add Caution Insulin | Medicinale caution (dementia dosing errors) |
| `48.json` | Update Middle Activity Breaking Dependency Chain | Update + violazione catena dipendenze |
| `49.json` | Add Time-Limited Intensive Physiotherapy | Validity period |
| `50.json` | Complex Add Medication and Social Activity | Add + conflitto scheduling + social |

## Patient 6 — Laura Bennett (Perimenopause, Mild Hypertension)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `51.json` | Add Safe Lisinopril | Medicinale sicuro |
| `52.json` | Add Evening Wine with History Warning | History warning (hot flashes) |
| `53.json` | Add Restaurant Meal with History Warning | History warning (BP elevation) |
| `54.json` | Add Yoga Relaxation Activity | Add semplice, relaxation |
| `55.json` | Add Activity Same Time Different Days - No Conflict | No conflitto (giorni diversi) |
| `56.json` | Update Time Causing Conflict - Accept Alternative | Update + conflitto + accept |
| `57.json` | Add Activity with Non-Existent Dependency | Dipendenza inesistente |
| `58.json` | Remove Activity Blocked by Dependent | Remove bloccato da dipendente |
| `59.json` | Add Activity with Non-Overlapping Validity - No Conflict | Validity period, no conflitto |
| `60.json` | Complex Add, Update, and Add with Dependency | Add + update + add + dipendenza |

## Patient 7 — Anne Morris (Hypertension, Type 2 Diabetes Mellitus)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `61.json` | Add Safe Metformin | Medicinale sicuro |
| `62.json` | Add Safe Lisinopril | Medicinale sicuro |
| `63.json` | Add Contraindicated Prednisone | Medicinale controindicato (diabete) |
| `64.json` | Add Safe Insulin with Dependency | Medicinale sicuro + dipendenza |
| `65.json` | Add Health Checkup - No History Warning | No history warning (info only) |
| `66.json` | Scheduling Conflict - Accept Alternative | Conflitto + accept |
| `67.json` | Add Activity with Valid Dependency | Dipendenza valida |
| `68.json` | Update Breaking Dependency Ordering | Violazione ordinamento dipendenza |
| `69.json` | Expired Activity Replacement with Validity Period | Expired + validity period |
| `70.json` | Complex Add, Query, and Add | Add + query interazione + add relaxation |

## Patient 8 — Joan Edwards (Rheumatoid Arthritis, Iron-Deficiency Anemia)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `71.json` | Add Safe Paracetamol | Medicinale sicuro |
| `72.json` | Add Safe Ferrous Sulfate | Medicinale sicuro |
| `73.json` | Add Contraindicated Ibuprofen | Medicinale controindicato (GI bleeding/anemia) |
| `74.json` | Add Caution Omeprazole | Medicinale caution (reduces iron absorption) |
| `75.json` | Add Caution Methotrexate | Medicinale caution (bone marrow suppression) |
| `76.json` | Add Caution Prednisone | Medicinale caution (osteoporosis risk) |
| `77.json` | Scheduling Conflict - Reject Alternative | Conflitto + reject + proposta propria |
| `78.json` | Add Therapy with Valid Dependency Chain and No History Warning | Catena dipendenze valida + no history warning (no history file) |
| `79.json` | Add Time-Limited Social Activity | Validity period + social |
| `80.json` | Complex Unknown Medicine Then Safe Add | Medicinale non in DB + safe add |

## Patient 9 — Rose Baker (Mild Alzheimer's Dementia)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `81.json` | Add Safe Donepezil | Medicinale sicuro |
| `82.json` | Add Contraindicated Warfarin | Medicinale controindicato (dementia fall/bleeding risk) |
| `83.json` | Add Overnight Travel with History Warning | History warning (disorientation) |
| `84.json` | Remove Evening Meal with History Warning | History warning (dizziness) + remove |
| `85.json` | Add Caution Rivaroxaban | Medicinale caution (fall risk remains) |
| `86.json` | Add Caution Insulin | Medicinale caution (dementia dosing errors) |
| `87.json` | Scheduling Conflict - Accept Alternative | Conflitto + accept |
| `88.json` | Update Breaking Dependency Ordering | Violazione ordinamento dipendenza |
| `89.json` | Add Activity with Non-Existent Dependency | Dipendenza inesistente |
| `90.json` | Complex Add, Update, and Remove | Add + update + remove combinati |

## Patient 10 — Carol Phillips (Heart Failure, Cognitive Decline)

| File | Scenario | Feature testata |
|------|----------|-----------------|
| `91.json` | Add Safe Furosemide | Medicinale sicuro |
| `92.json` | Add Safe Lisinopril | Medicinale sicuro |
| `93.json` | Add Contraindicated Metformin | Medicinale controindicato (lactic acidosis/heart failure) |
| `94.json` | Contraindicated Propranolol, Safe Paracetamol, and Supervised Walking | Medicinale controindicato (Propranolol/HF) + medicinale sicuro (Paracetamol) + no history warning (info only) |
| `95.json` | Add Contraindicated Aspirin | Medicinale controindicato (heart failure) |
| `96.json` | Add Contraindicated Warfarin | Medicinale controindicato (cognitive decline fall/bleeding risk) |
| `97.json` | Add Contraindicated Ibuprofen | Medicinale controindicato (heart failure) |
| `98.json` | Remove Activity Blocked by Dependent | Remove bloccato da dipendente |
| `99.json` | Expired Activity Replacement with Validity Period | Expired + validity period |
| `100.json` | Complex Add, Update, Add with Dependency, and Caution Medicine | Add + update + add + dipendenza + medicinale caution (Rivaroxaban) |

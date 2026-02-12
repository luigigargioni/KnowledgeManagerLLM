import json
import logging
from pathlib import Path
from datetime import datetime
import requests

logger = logging.getLogger("__main__")

THERAPY_FILE = Path("../data/therapy.json")





def get_devices():
    data=requests.get("http://192.168.1.118:8000/device?get_only_names=true")
    if(data):
        return data.text
    else:
        return "Nessun device trovato"
    
def clear_conversation_history(self,keep_system=True):
    if keep_system and self.conversation_history and self.conversation_history[0]['role'] == 'system':
        system_msg = self.conversation_history[0]
        self.conversation_history = [system_msg]
        return "Cronologia pulita. System prompt mantenuto."
    else:
        self.conversation_history = []
        return "Cronologia completamente pulita."


def _ensure_data_dir():
    """Crea la cartella data se non esiste"""
    THERAPY_FILE.parent.mkdir(exist_ok=True)


def _load_therapy():
    """Carica il file therapy.json"""
    _ensure_data_dir()
    
    if not THERAPY_FILE.exists():
        logger.warning("[THERAPY] therapy.json not found, creating empty structure")
        default_data = {
            "patient_id": "test",
            "patient_name": "Test",
            "activities": []
        }
        _save_therapy(default_data)
        return default_data
    
    try:
        with open(THERAPY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[THERAPY] Error loading therapy.json: {e}")
        raise


def _save_therapy(data):
    """Salva il file therapy.json"""
    _ensure_data_dir()
    
    try:
        with open(THERAPY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.debug(f"[THERAPY] Saved therapy data: {len(data.get('activities', []))} activities")
    except Exception as e:
        logger.error(f"[THERAPY] Error saving therapy.json: {e}")
        raise


def get_all_activities():
    """
    Ottiene tutte le attività terapeutiche
    
    Returns:
        str: JSON formattato con tutte le attività
    """
    try:
        data = _load_therapy()
        
        if not data.get('activities'):
            return json.dumps({
                "status": "success",
                "message": "Nessuna attività configurata",
                "patient_id": data.get('patient_id', ''),
                "patient_name": data.get('patient_name', ''),
                "activities": []
            }, indent=2, ensure_ascii=False)
        
        logger.info(f"[THERAPY] Retrieved {len(data['activities'])} activities")
        data.update({"status": "success",})
        return json.dumps(data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[THERAPY] Error getting activities: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def add_activity(activity_data):
    """
    Aggiunge una nuova attività terapeutica
    
    Args:
        activity_data: dict con i dati dell'attività (activity_id, name, description, day_of_week, time, duration_minutes, dependencies, valid_from, valid_until)
    
    Returns:
        str: Messaggio di conferma o errore
    """
    try:
        data = _load_therapy()
        
        # Validazione campi obbligatori
        required_fields = ['activity_id', 'name', 'day_of_week', 'time', 'duration_minutes']
        for field in required_fields:
            if field not in activity_data:
                return json.dumps({
                    "status": "error",
                    "message": f"Campo obbligatorio mancante: {field}"
                }, indent=2)
        
        # Verifica che activity_id non esista già
        if any(act['activity_id'] == activity_data['activity_id'] for act in data['activities']):
            return json.dumps({
                "status": "error",
                "message": f"Attività con ID '{activity_data['activity_id']}' già esistente"
            }, indent=2)
        
        # Imposta valori di default
        new_activity = {
            "activity_id": activity_data['activity_id'],
            "name": activity_data['name'],
            "description": activity_data.get('description', ''),
            "day_of_week": activity_data['day_of_week'],
            "time": activity_data['time'],
            "duration_minutes": activity_data['duration_minutes'],
            "dependencies": activity_data.get('dependencies', []),
            "valid_from": activity_data.get('valid_from'),
            "valid_until": activity_data.get('valid_until')
        }
        
        data['activities'].append(new_activity)
        _save_therapy(data)
        
        logger.info(f"[THERAPY] Added activity: {new_activity['activity_id']} - {new_activity['name']}")
        
        return json.dumps({
            "status": "success",
            "message": f"Attività '{new_activity['name']}' aggiunta con successo",
            "activity": new_activity
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[THERAPY] Error adding activity: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def update_activity(activity_id, updates):
    """
    Aggiorna un'attività esistente
    
    Args:
        activity_id: ID dell'attività da aggiornare
        updates: dict con i campi da aggiornare
    
    Returns:
        str: Messaggio di conferma o errore
    """
    try:
        data = _load_therapy()
        
        # Trova l'attività
        activity_index = None
        for i, act in enumerate(data['activities']):
            if act['activity_id'] == activity_id:
                activity_index = i
                break
        
        if activity_index is None:
            return json.dumps({
                "status": "error",
                "message": f"Attività con ID '{activity_id}' non trovata"
            }, indent=2)
        
        # Aggiorna i campi
        activity = data['activities'][activity_index]
        old_activity = activity.copy()
        
        for key, value in updates.items():
            if key != 'activity_id':  # Non permettere di cambiare l'ID
                activity[key] = value
        
        _save_therapy(data)
        
        logger.info(f"[THERAPY] Updated activity: {activity_id}")
        logger.debug(f"[THERAPY] Old: {old_activity}")
        logger.debug(f"[THERAPY] New: {activity}")
        
        return json.dumps({
            "status": "success",
            "message": f"Attività '{activity['name']}' aggiornata con successo",
            "activity": activity
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[THERAPY] Error updating activity: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def remove_activity(activity_id):
    """
    Rimuove un'attività terapeutica
    
    Args:
        activity_id: ID dell'attività da rimuovere
    
    Returns:
        str: Messaggio di conferma o errore
    """
    try:
        data = _load_therapy()
        
        # Trova e rimuovi l'attività
        activity_index = None
        removed_activity = None
        
        for i, act in enumerate(data['activities']):
            if act['activity_id'] == activity_id:
                activity_index = i
                removed_activity = act
                break
        
        if activity_index is None:
            return json.dumps({
                "status": "error",
                "message": f"Attività con ID '{activity_id}' non trovata"
            }, indent=2)
        
        data['activities'].pop(activity_index)
        _save_therapy(data)
        
        logger.info(f"[THERAPY] Removed activity: {activity_id} - {removed_activity['name']}")
        
        return json.dumps({
            "status": "success",
            "message": f"Attività '{removed_activity['name']}' rimossa con successo",
            "removed_activity": removed_activity
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[THERAPY] Error removing activity: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)










#region Tools declaration
tools_decl=[
            {
                "type": "function",
                "function": {
                    "name": "get_devices",
                    "description": "Ottiene la lista dei dispositivi della tua smart home ",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
                        {
                "type": "function",
                "function": {
                    "name": "add_activity",
                    "description": "Aggiunge un'attività nella terapia del paziente corrente ",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
                        {
                "type": "function",
                "function": {
                    "name": "update_activity",
                    "description": "Aggiorna un'attività nella terapia del paziente corrente",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
                        {
                "type": "function",
                "function": {
                    "name": "delete_activity",
                    "description": "Rimuove un'attività nella terapia del paziente corrente",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_datetime",
                    "description": "Ottiene la data e l'ora corrente nel formato leggibile",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_conversation_history",
                    "description": "Pulisce tutta la cronologia della conversazione. Usa questo quando l'utente vuole ricominciare da capo o resettare la chat",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keep_system_prompt": {
                                "type": "boolean",
                                "description": "Se true, mantiene il system prompt anche dopo aver pulito la cronologia"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_therapy_activities",
                    "description": "Ottiene tutte le attività terapeutiche configurate per il paziente",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_therapy_activity",
                    "description": "Aggiunge una nuova attività terapeutica. Richiede: activity_id (unico), name, day_of_week (lista di giorni 0-6, dove 0=lunedì), time (formato HH:MM), duration_minutes. Opzionali: description, dependencies (lista nomi attività), valid_from, valid_until",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "activity_id": {
                                "type": "string",
                                "description": "ID univoco dell'attività (es: 'lb_001')"
                            },
                            "name": {
                                "type": "string",
                                "description": "Nome dell'attività"
                            },
                            "description": {
                                "type": "string",
                                "description": "Descrizione dettagliata dell'attività"
                            },
                            "day_of_week": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Giorni della settimana (0=lunedì, 6=domenica)"
                            },
                            "time": {
                                "type": "string",
                                "description": "Orario dell'attività (formato HH:MM)"
                            },
                            "duration_minutes": {
                                "type": "integer",
                                "description": "Durata in minuti"
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Lista di nomi di attività da completare prima"
                            },
                            "valid_from": {
                                "type": "string",
                                "description": "Data di inizio validità (YYYY-MM-DD)"
                            },
                            "valid_until": {
                                "type": "string",
                                "description": "Data di fine validità (YYYY-MM-DD)"
                            }
                        },
                        "required": ["activity_id", "name", "day_of_week", "time", "duration_minutes"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_therapy_activity",
                    "description": "Aggiorna un'attività terapeutica esistente. Specifica l'activity_id e i campi da modificare",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "activity_id": {
                                "type": "string",
                                "description": "ID dell'attività da aggiornare"
                            },
                            "updates": {
                                "type": "object",
                                "description": "Oggetto con i campi da aggiornare (name, description, day_of_week, time, duration_minutes, dependencies, valid_from, valid_until)"
                            }
                        },
                        "required": ["activity_id", "updates"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_therapy_activity",
                    "description": "Rimuove un'attività terapeutica",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "activity_id": {
                                "type": "string",
                                "description": "ID dell'attività da rimuovere"
                            }
                        },
                        "required": ["activity_id"]
                    }
                }
            },
        ]
        
#endregion
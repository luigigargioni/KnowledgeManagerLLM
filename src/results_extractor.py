# results_excel.py

import json
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config_loader import RESULTS_DIR

RESULTS_EXCEL_PATH = RESULTS_DIR / "all_results.xlsx"

# Colonne del foglio principale (una riga per obiettivo)
_OBJECTIVE_COLUMNS = [
    "test_date",
    "batch_id",
    "scenario_id",
    "patient",
    "overall_status",
    "turns",
    "elapsed_seconds",
    "objectives",
    "judge_check",
    "conversation",
    "final_therapy",
]

# Colori per i vari status
_STATUS_FILLS = {
    "completed": PatternFill("solid", fgColor="C6EFCE"),  # verde
    "partial": PatternFill("solid", fgColor="FFEB9C"),  # giallo
    "failed": PatternFill("solid", fgColor="FFC7CE"),  # rosso
    "not_attempted": PatternFill("solid", fgColor="D9D9D9"),  # grigio
    "error": PatternFill("solid", fgColor="F4CCCC"),  # rosso scuro
}

_HEADER_FILL = PatternFill("solid", fgColor="2F4F7F")
_HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=10)
_CELL_FONT = Font(name="Arial", size=10)
_BORDER_SIDE = Side(style="thin", color="CCCCCC")
_THIN_BORDER = Border(
    left=_BORDER_SIDE,
    right=_BORDER_SIDE,
    top=_BORDER_SIDE,
    bottom=_BORDER_SIDE,
)


def _get_or_create_workbook() -> tuple[openpyxl.Workbook, bool]:
    """
    Carica il workbook esistente o ne crea uno nuovo.
    Restituisce (workbook, is_new).
    """
    if RESULTS_EXCEL_PATH.exists():
        return openpyxl.load_workbook(RESULTS_EXCEL_PATH), False

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # rimuove il foglio default vuoto
    return wb, True


def _get_or_create_sheet(wb: openpyxl.Workbook, sheet_name: str, is_new_wb: bool):
    """
    Restituisce il foglio esistente o ne crea uno nuovo con intestazioni.
    """
    if sheet_name in wb.sheetnames:
        return wb[sheet_name]

    ws = wb.create_sheet(sheet_name)
    _write_headers(ws, _OBJECTIVE_COLUMNS)
    return ws


def _write_headers(ws, columns: list[str]) -> None:
    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.border = _THIN_BORDER
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )


def _style_cell(cell, status: str | None = None) -> None:
    cell.font = _CELL_FONT
    cell.border = _THIN_BORDER
    cell.alignment = Alignment(vertical="top", wrap_text=True)
    if status and status in _STATUS_FILLS:
        cell.fill = _STATUS_FILLS[status]


def _set_column_widths(ws, columns: list[str]) -> None:
    widths = {
        "test_date": 18,
        "batch_id": 20,
        "scenario_id": 12,
        "patient": 22,
        "overall_status": 16,
        "turns": 8,
        "elapsed_seconds": 16,
        "judge_check": 50,
        "objectives": 50,
        "conversation": 80,
        "final_therapy": 50,
    }
    for col_idx, col_name in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = widths.get(
            col_name, 20
        )


def append_batch_results(
    evaluation: dict,
    batch_id: str,
    scenario: str,
    conversation: str,
    final_therapy: dict,
    excel_path: Path = RESULTS_EXCEL_PATH,
) -> Path:
    """
    Aggiunge i risultati di un batch run al file Excel globale.
    Se il file non esiste viene creato. Se esiste, le righe vengono
    appese preservando i dati delle esecuzioni precedenti.

    Ogni obiettivo di ogni scenario occupa una riga separata.
    Le colonne scenario_id e test_date identificano univocamente l'esecuzione.

    Args:
        evaluations: lista di dict prodotti da JudgeAgent.evaluate(),
                     arricchiti da test.py con scenario_id, patient, turns, elapsed_seconds
        batch_id:    identificatore del batch (es. "20260630_143000")
        excel_path:  path del file Excel globale (default: logs/batch_results/all_results.xlsx)

    Returns:
        Path del file Excel aggiornato.
    """
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    wb, is_new_wb = _get_or_create_workbook()
    ws = _get_or_create_sheet(wb, "Results", is_new_wb)

    if is_new_wb or "Results" not in wb.sheetnames or ws.max_row == 1:
        _set_column_widths(ws, _OBJECTIVE_COLUMNS)

    test_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if evaluation.get("status") == "error":
        # Scenario fallito: scrivi una riga singola con lo stato di errore
        row = [
            test_date,
            batch_id,
            evaluation.get("scenario_id", ""),
            evaluation.get("patient", ""),
            "error",
            evaluation.get("turns", ""),
            evaluation.get("elapsed_seconds", ""),
            scenario,
            evaluation.get("message", "Evaluation failed"),
            evaluation.get("raw_output", "")[:500],
            conversation or "",
            "",
        ]
        ws.append(row)
        row_idx = ws.max_row
        for col_idx in range(1, len(_OBJECTIVE_COLUMNS) + 1):
            _style_cell(ws.cell(row_idx, col_idx), "error")

    else:
        scenario_id = evaluation.get("scenario_id", "")
        patient = evaluation.get("patient", "")
        overall_status = evaluation.get("overall_status", "")
        turns = evaluation.get("turns", "")
        elapsed = evaluation.get("elapsed_seconds", "")
        objectives = evaluation.get("objectives", [])

        row = [
            test_date,
            batch_id,
            scenario_id,
            patient,
            overall_status,
            turns,
            elapsed,
            scenario,
            json.dumps(objectives, indent=2),
            conversation,
            json.dumps(final_therapy, indent=2),
        ]
        ws.append(row)
        row_idx = ws.max_row
        for col_idx in range(1, len(_OBJECTIVE_COLUMNS) + 1):
            _style_cell(ws.cell(row_idx, col_idx), overall_status)
    # Freezes la riga di intestazione
    ws.freeze_panes = "A2"

    wb.save(excel_path)
    return excel_path

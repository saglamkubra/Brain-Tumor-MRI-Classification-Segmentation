import json
from pathlib import Path

ENGLISH = json.loads((Path(__file__).resolve().parents[1] / "frontend" / "messages.json").read_text(encoding="utf-8"))


def message(value, language):
    if language != "en" or not isinstance(value, str):
        return value
    if value in ENGLISH:
        return ENGLISH[value]
    if value.startswith("Aşama ") and ": " in value:
        stage, reason = value.split(": ", 1)
        return f"Stage {stage.split()[-1]}: {ENGLISH.get(reason.rstrip('.'), reason.rstrip('.'))}."
    return value

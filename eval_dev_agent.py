import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
BENCHMARK = BASE / "starter/data/converted/questions_full.json"
OUT = BASE / "starter/results/agent-dev"

OUT.mkdir(parents=True, exist_ok=True)

questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))
dev = [q for q in questions if q["split"] == "dev"]

assert len(dev) == 17, f"Expected 17 DEV, found {len(dev)}"

def field(pattern, value):
    m = re.search(pattern, value)
    return m.group(1).strip() if m else None

results = []

print("=== GEMINI AGENT DEV EVALUATION ===", flush=True)
print("Questions:", len(dev), flush=True)

for index, q in enumerate(dev, 1):
    qid = q["id"]
    print(f"\n[{index}/{len(dev)}] {qid}", flush=True)

    command = [
        sys.executable,
        str(BASE / "run_gemini_agent.py"),
        "--qid", qid
    ]

    error = None

    try:
        completed = subprocess.run(
            command,
            cwd=BASE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120
        )
        output = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0:
            error = f"exit_code_{completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        output = (
            str(exc.stdout or b"") + "\n" +
            str(exc.stderr or b"")
        )
        error = "timeout"

    (OUT / f"{qid}.log").write_text(output, encoding="utf-8")

    actions = []
    for line in output.splitlines():
        if line.startswith("NORMALIZED ACTION:"):
            try:
                actions.append(json.loads(
                    line.split("NORMALIZED ACTION:", 1)[1].strip()
                ))
            except json.JSONDecodeError:
                pass

    raw_status = field(r"(?m)^Status:\s*(.+)$", output)
    events = re.findall(r"(?m)^Event:\s*(\S+)", output)
    last_action = actions[-1] if actions else {}

    if error:
        classification = "api_or_runtime_error"
    elif (
        last_action.get("action") == "answer"
        and last_action.get("answer") == ""
        and last_action.get("citations") == []
    ):
        classification = "intentional_abstain"
    elif raw_status == "answered":
        classification = "answered"
    elif "search_limit" in events:
        classification = "search_limit"
    elif "invalid_or_missing_citations" in events:
        classification = "invalid_citation_or_answer"
    else:
        classification = "other"

    record = {
        "id": qid,
        "question": q["question"],
        "answerable": q.get("answerable"),
        "expected_behavior": q.get("expected_behavior"),
        "status": raw_status,
        "classification": classification,
        "events": events,
        "model_calls": field(r"(?m)^Model calls:\s*(\d+)", output),
        "elapsed_ms": field(r"(?m)^Elapsed ms:\s*([\d.]+)", output),
        "error": error,
        "log": str(OUT / f"{qid}.log")
    }

    results.append(record)

    print("Status:", raw_status, flush=True)
    print("Classification:", classification, flush=True)

    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

from collections import Counter

summary = {
    "created_at": datetime.now().isoformat(),
    "split": "dev",
    "total": len(results),
    "classifications": dict(
        Counter(r["classification"] for r in results)
    ),
    "note": "Behavioral summary only; semantic grounding requires human review."
}

(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== DEV SUMMARY ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Results:", OUT / "results.json")

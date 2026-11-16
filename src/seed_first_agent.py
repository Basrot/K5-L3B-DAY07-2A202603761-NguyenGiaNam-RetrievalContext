"""Seed-first context expansion for Lab 07."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "starter/data/converted/chunks.json"


def load_chunks():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return {c["id"]: c for c in chunks}


def find_neighbors(cid, by_id):
    match = re.fullmatch(r"(.+_para)(\d+)", cid)
    if not match or cid not in by_id:
        return []

    prefix = match.group(1)
    number = int(match.group(2))
    current = by_id[cid]

    output = []

    for index in (number - 1, number + 1):
        if index < 0:
            continue

        neighbor_id = f"{prefix}{index}"
        neighbor = by_id.get(neighbor_id)

        if (
            neighbor
            and neighbor.get("source_id") == current.get("source_id")
            and neighbor.get("page") == current.get("page")
        ):
            output.append(neighbor_id)

    return output


def pack_seed_first(candidates, by_id, budget):
    seeds = list(dict.fromkeys(
        c["id"] for c in candidates
    ))

    selected = []
    blocks = []
    used = 0

    def try_add(cid):
        nonlocal used

        if cid in selected or cid not in by_id:
            return

        c = by_id[cid]
        block = (
            f"[{cid} | {c.get('source', '')} | "
            f"page {c.get('page')}]\n"
            f"{c['text']}\n\n"
        )

        if used + len(block) <= budget:
            selected.append(cid)
            blocks.append(block)
            used += len(block)

    # Phase 1: preserve ranked seeds.
    for cid in seeds:
        try_add(cid)

    # Phase 2: use remaining budget for neighbors.
    for cid in seeds:
        for neighbor_id in find_neighbors(cid, by_id):
            try_add(neighbor_id)

    return "".join(blocks), selected


def run_agent_seedfirst(
    question,
    controller,
    retriever,
    *,
    filters=None,
    max_searches=3,
    top_k=3,
    budget=2400,
):
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("Invalid agent parameters")

    by_id = load_chunks()
    history = []
    candidates = []
    events = []
    visible_chunks = {}

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)

        if action.get("action") == "search":
            if sum(e["event"] == "search" for e in events) >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}],
                }

            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Invalid search query")

            results = retriever.search(
                query, filters=filters, top_k=top_k
            )

            if (
                len(results) > top_k
                or len({r["id"] for r in results}) != len(results)
            ):
                raise ValueError("Invalid retriever results")

            candidates.extend(results)

            context, ids = pack_seed_first(
                candidates, by_id, budget
            )

            visible_chunks = {
                cid: by_id[cid] for cid in ids
            }

            events.append({
                "event": "search",
                "query": query,
                "filter": filters,
                "results": results,
                "context": context,
                "context_ids": ids,
            })

            history = [{
                "tool": "search",
                "queries": [
                    e["query"] for e in events
                    if e["event"] == "search"
                ],
                "context": context,
                "context_ids": ids,
            }]

        elif action.get("action") == "answer":
            answer = action.get("answer")
            citations = action.get("citations")

            if (
                not isinstance(answer, str)
                or not isinstance(citations, list)
                or any(not isinstance(c, str) for c in citations)
            ):
                raise ValueError("Invalid answer/citations")

            if not answer.strip() and not citations:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [
                        {"event": "intentional_abstain"}
                    ],
                }

            valid = bool(
                answer.strip()
                and citations
                and all(
                    cid in visible_chunks and f"[{cid}]" in answer
                    for cid in citations
                )
            )

            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "events": events + [{"event": "answer"}],
                }

            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [
                    {"event": "invalid_or_missing_citations"}
                ],
            }

        else:
            raise ValueError("Unknown agent action")

    raise RuntimeError("Agent loop ended unexpectedly")

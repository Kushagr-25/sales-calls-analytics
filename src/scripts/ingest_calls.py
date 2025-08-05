"""
Async ingestion of JSONL + raw transcripts.
Run:  python -m src.scripts.ingest_calls  (from repo root)
"""

import asyncio
import json
import re
from pathlib import Path
from typing import AsyncIterator, Dict

import aiofiles  # type: ignore[import]
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src/dataset/sales_call_dataset"
JSONL_PATH = DATA_DIR / "calls.jsonl"
RAW_DIR = DATA_DIR / "raw"

PROMPT_HEADER_RE = re.compile(r"^\s*You are simulating.*$", re.I)

# ---------- helpers --------------------------------------------------


async def stream_jsonl(fp: Path) -> AsyncIterator[Dict]:
    """Yield dicts from a JSON Lines file asynchronously."""
    async with aiofiles.open(fp, "r", encoding="utf-8") as f:
        async for line in f:
            yield json.loads(line)


async def load_transcript(call_id: str) -> str:
    txt_path = RAW_DIR / f"{call_id}.txt"
    async with aiofiles.open(txt_path, "r", encoding="utf-8") as f:
        raw = await f.read()

    # strip optional first line prompt header
    cleaned_lines = [ln for ln in raw.splitlines() if not PROMPT_HEADER_RE.match(ln)]
    return "\n".join(cleaned_lines).strip()


# ---------- main -----------------------------------------------------


async def ingest() -> None:
    async for record in atqdm(stream_jsonl(JSONL_PATH), total=200):
        call_id = record["call_id"]

        try:
            transcript = await load_transcript(call_id)
        except FileNotFoundError:
            print(f"[WARN] transcript file missing for {call_id}")
            continue

        # minimal normalisation (Phase 2 only)
        parsed = {
            **record,  # call_id, agent_id, …
            "transcript": transcript,  # cleaned
        }

        # For Phase 2 we just echo; Phase 3 will bulk-insert here
        print(json.dumps(parsed)[:120] + "…")  # preview first 120 chars


def main() -> None:
    asyncio.run(ingest())


if __name__ == "__main__":
    main()

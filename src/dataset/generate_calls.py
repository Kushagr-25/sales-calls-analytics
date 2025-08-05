"""
Generate synthetic sales-call transcripts and metadata.

Usage:
    python generate_calls.py --n_calls 250 --out_dir data --local
"""

import datetime as dt
import json
import os
import random
import time
import uuid
from pathlib import Path

import click
from dateutil import tz  # type: ignore[import]
from faker import Faker
from tqdm import tqdm

# -------- Hugging Face / LLM helpers ------------------
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def load_generator(model_name: str, local: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if local:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        return pipeline("text-generation", model=mod, tokenizer=tok)
    else:  # HF serverless
        from transformers import pipeline

        return pipeline("text-generation", model=model_name)


def craft_prompt(agent, customer, product, language, turns=12):
    sys_msg = (
        f"You are simulating a realistic sales phone call in {language}. "
        "Write a conversation transcript with clear speaker labels:"
        f"\nAgent ({agent}) and Customer ({customer}). "
        f"Product: {product}. Keep it helpful, natural, and around {turns} exchanges."
    )
    return sys_msg


# -------- Dataset generation --------------------------
fake = Faker()
Faker.seed(2025)
random.seed(2025)

LANGS = ["English", "Hindi", "Spanish", "French", "German"]
PRODUCTS = [
    "SaaS CRM subscription",
    "enterprise cloud backup",
    "AI-powered chatbot plan",
    "premium credit card",
    "solar panel installation",
    "health-tech wearable",
]


def single_call(gen, local):
    call_id = str(uuid.uuid4())
    agent_id = fake.unique.user_name()
    customer_id = fake.unique.user_name()
    language = random.choice(LANGS)
    start_time = fake.date_time_between(
        start_date="-90d", end_date="now", tzinfo=tz.gettz("UTC")
    ).isoformat()
    duration_seconds = random.randint(90, 600)

    prompt = craft_prompt(agent_id, customer_id, random.choice(PRODUCTS), language)
    result = gen(prompt, max_new_tokens=256, temperature=0.8)[0]["generated_text"]

    return {
        "call_id": call_id,
        "agent_id": agent_id,
        "customer_id": customer_id,
        "language": language,
        "start_time": start_time,
        "duration_seconds": duration_seconds,
        "transcript": result.strip(),
    }


# -------- CLI entry-point -----------------------------
@click.command()
@click.option("--n_calls", default=200, help="Number of transcripts to generate.")
@click.option("--out_dir", default="data", help="Root output directory.")
@click.option("--model", default=DEFAULT_MODEL, help="HF model ID to use.")
@click.option(
    "--local/--remote",
    default=True,
    help="Use local weights (True) or HuggingFace Inference API (False).",
)
def main(n_calls, out_dir, model, local):
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    gen = load_generator(model, local)

    with open(out_dir / "calls.jsonl", "w", encoding="utf-8") as fout:
        for _ in tqdm(range(n_calls), desc="Generating calls"):
            call = single_call(gen, local)
            # save raw transcript
            with open(
                raw_dir / f"{call['call_id']}.txt", "w", encoding="utf-8"
            ) as fraw:
                fraw.write(call["transcript"])
            # write normalised row
            fout.write(json.dumps(call, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

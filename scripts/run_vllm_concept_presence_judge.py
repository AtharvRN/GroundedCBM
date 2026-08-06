#!/usr/bin/env python3
"""Run concept-presence VLM judging against a local OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


_write_lock = threading.Lock()
_thread_local = threading.local()


def get_client(base_url: str, api_key: str) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        _thread_local.client = client
    return client


def read_completed(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = row.get("task_id")
            if task_id:
                completed.add(task_id)
    return completed


def load_tasks(input_path: Path, completed: set[str], limit: int | None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") in completed:
                continue
            tasks.append(row)
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value, None
        return None, "decoded JSON was not an object"
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return value, None
            return None, "extracted JSON was not an object"
        except json.JSONDecodeError as exc:
            return None, f"failed to decode extracted JSON: {exc}"
    return None, "no JSON object found"


def judge_one(
    row: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    image_path = Path(row["image_file"])
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = row["prompt_template"]

    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            client = get_client(base_url, api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            parsed, parse_error = parse_json_object(raw)
            result = {
                "task_id": row.get("task_id"),
                "dataset_index": row.get("dataset_index"),
                "model_name": row.get("model_name"),
                "concept_name": row.get("concept_name"),
                "image_file": row.get("image_file"),
                "metadata": row.get("metadata", {}),
                "judge_model": model,
                "raw_response": raw,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "ok": parsed is not None and parse_error is None,
            }
            return result
        except Exception as exc:  # noqa: BLE001 - persist the failure for resume/debugging.
            last_error = repr(exc)
            time.sleep(min(2**attempt, 10))

    return {
        "task_id": row.get("task_id"),
        "dataset_index": row.get("dataset_index"),
        "model_name": row.get("model_name"),
        "concept_name": row.get("concept_name"),
        "image_file": row.get("image_file"),
        "metadata": row.get("metadata", {}),
        "judge_model": model,
        "raw_response": "",
        "parsed_response": None,
        "parse_error": last_error,
        "ok": False,
    }


def process_file(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    output_path = input_path.parent / args.output_name
    completed = read_completed(output_path)
    tasks = load_tasks(input_path, completed, args.limit_per_file)
    started = time.time()
    written = 0
    failed = 0

    with output_path.open("a", encoding="utf-8") as out:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = [
                executor.submit(
                    judge_one,
                    task,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                )
                for task in tasks
            ]
            for future in futures.as_completed(pending):
                result = future.result()
                with _write_lock:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                written += 1
                failed += int(not result.get("ok"))
                if written % args.log_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "file": str(input_path),
                                "written": written,
                                "remaining_in_file": len(tasks) - written,
                                "failed": failed,
                                "rate_tasks_per_min": written / elapsed * 60.0,
                            }
                        ),
                        flush=True,
                    )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "already_completed": len(completed),
        "new_written": written,
        "failed": failed,
        "elapsed_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*/judge_tasks.jsonl")
    parser.add_argument("--output-name", default="vlm_judge_qwen25vl7b.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen25vl7b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit-per-file", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    input_files = sorted(args.input_root.glob(args.pattern))
    if args.only:
        wanted = set(args.only)
        input_files = [p for p in input_files if p.parent.name in wanted]
    if not input_files:
        raise SystemExit(f"No input files matched {args.input_root / args.pattern}")

    summaries = []
    for input_path in input_files:
        print(json.dumps({"event": "start_file", "input": str(input_path)}), flush=True)
        summary = process_file(args, input_path)
        summaries.append(summary)
        print(json.dumps({"event": "finish_file", **summary}), flush=True)

    print(json.dumps({"event": "done", "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()

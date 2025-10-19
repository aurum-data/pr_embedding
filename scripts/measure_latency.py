from __future__ import annotations

"""
Latency probe for the Prime Radiant embedding API.

Run `python scripts/measure_latency.py --help` for a full list of flags and usage examples.
"""

import argparse
import statistics
import time
from typing import Sequence

import httpx

# Default configuration values. Feel free to override via CLI flags.
DEFAULT_BASE_URL = "https://pr-embedding.onrender.com"
DEFAULT_COUNT    = 5
DEFAULT_WARMUP   = 1
DEFAULT_TIMEOUT  = 120.0
DEFAULT_INPUTS   = [
    "Prime Radiant latency check.",
    "Nomic embed text via llama.cpp.",
]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser so users can customize the probe."""
    parser = argparse.ArgumentParser(
        description=(
            "Send repeated embedding requests to the Prime Radiant server and "
            "report per-call latency statistics."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL where the embedding service is listening.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of timed embedding requests to send after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of warmup requests to send before measuring latency.",
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=None,
        help=(
            "An input string to embed. Repeat the flag to send multiple strings "
            "per request. Defaults to a single canned prompt."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size used by the embedding call.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle output vector normalization (default: enabled).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP client timeout in seconds.",
    )
    return parser


def ensure_list(inputs: Sequence[str] | None) -> list[str]:
    """
    Guarantee the payload contains at least one string to embed.

    The server accepts both a single string and a list; we swap to the simplest default
    benchmark prompts when nothing is supplied via the CLI.
    """
    if not inputs:
        return DEFAULT_INPUTS.copy()
    return list(inputs)


def run_trial(
    client: httpx.Client,
    url: str,
    payload: dict,
) -> float:
    """
    Execute a single POST to the embeddings endpoint and return elapsed seconds.

    The timer wraps both network time and server-side processing latency.
    """
    start = time.perf_counter()
    response = client.post(url, json=payload)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return elapsed


def format_ms(value: float) -> str:
    """Convert seconds to a human-friendly millisecond string."""
    return f"{value * 1_000:.1f} ms"


def main() -> None:
    """Parse CLI arguments, warm the model, then capture latency statistics."""
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    health_url = f"{base_url}/health"
    embeddings_url = f"{base_url}/v1/embeddings"
    inputs = ensure_list(args.inputs)

    # The API accepts either a single string or a list, so keep the payload compact.
    payload = {
        "input": inputs if len(inputs) > 1 else inputs[0],
        "normalize": args.normalize,
    }
    if args.batch_size:
        payload["batch_size"] = args.batch_size

    with httpx.Client(timeout=args.timeout) as client:
        # Probe /health to confirm the service is reachable before timing requests.
        try:
            health = client.get(health_url)
            health.raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                "Health check failed. Ensure the embedding server is running and the "
                "base URL is correct."
            ) from exc

        model_name = health.json().get("model", "unknown")
        print(
            f"Health check OK — model: {model_name}. "
            "Tip: run with --help to see more options."
        )

        # Warmups get the model into RAM before the timed trials.
        for index in range(args.warmup):
            run_trial(client, embeddings_url, payload)
            print(f"Warmup {index + 1}/{args.warmup} complete.")

        timings: list[float] = []
        for attempt in range(args.count):
            elapsed = run_trial(client, embeddings_url, payload)
            timings.append(elapsed)
            print(f"Trial {attempt + 1}/{args.count}: {format_ms(elapsed)}")

    if not timings:
        print("No timings collected.")
        return

    # Summarize the captured latency profile so we can spot outliers quickly.
    mean = statistics.mean(timings)
    median = statistics.median(timings)
    fastest = min(timings)
    slowest = max(timings)

    print("\nLatency summary (wall clock):")
    print(f"  mean   : {format_ms(mean)}")
    print(f"  median : {format_ms(median)}")
    print(f"  fastest: {format_ms(fastest)}")
    print(f"  slowest: {format_ms(slowest)}")


if __name__ == "__main__":
    # Allow the script to be invoked directly from the CLI.
    main()

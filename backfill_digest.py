"""Backfill daily digests over a date range (standalone script).

Use this when the scheduler was stopped for a while and you need to regenerate
the missing digests. Every day in the range runs with force_update=True, so all
papers are re-summarized and existing DB entries are overwritten.

Each day calls run_daily_digest(...) exactly like the scheduler does — the
output .md lands in summaries/<YYYY>/<MM>/ and is uploaded to MinIO by the
graph. After the whole range finishes it triggers a single website rebuild via
trigger_deploy() (non-fatal; skipped if GITHUB_TOKEN is unset).

Usage:
    python backfill_digest.py                      # default range 30/05 -> 26/06
    python backfill_digest.py --start 2026-05-30 --end 2026-06-26
    python backfill_digest.py --model kimi/kimi-k2.6
    python backfill_digest.py --no-deploy          # skip the final web rebuild

Note: this is slow — one day already takes a long time, so 28 days will run for
many hours. A failure on a single day is logged but does not stop the rest.
"""
import os
import sys
import argparse
import logging
from datetime import date, datetime, timedelta

from dotenv import load_dotenv

load_dotenv(override=False)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure the log directory exists before configuring the file handler.
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/backfill.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

# Imported after logging is configured so its log lines are captured.
from daily_papers_tool import run_daily_digest  # noqa: E402

# Default range: the gap left by the paused scheduler.
DEFAULT_START = "2026-05-30"
DEFAULT_END = "2026-06-26"


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill daily digests over a date range (force update)."
    )
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help="Start date YYYY-MM-DD (default: 2026-05-30)")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help="End date YYYY-MM-DD inclusive (default: 2026-06-26)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model ID (default: from .env LLM_MODEL)")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Do not trigger the website rebuild at the end")
    args = parser.parse_args()

    model = args.model or os.getenv('LLM_MODEL', 'moonshotai/kimi-k2.5')
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info("=" * 60)
    logger.info(f"Backfill {start} -> {end} | model={model} | force_update=True")
    logger.info("=" * 60)

    ok = 0
    failed = 0
    for d in daterange(start, end):
        ds = d.strftime("%Y-%m-%d")
        logger.info(f"--- {ds} ---")
        try:
            result = run_daily_digest(ds, model=model, force_update=True)
            if result:
                ok += 1
                logger.info(f"{ds}: OK -> {result}")
            else:
                logger.info(f"{ds}: no papers found, skipped")
        except Exception as e:  # noqa: BLE001 — keep going on a bad day
            failed += 1
            logger.error(f"{ds}: FAILED ({e})", exc_info=True)

    logger.info("=" * 60)
    logger.info(f"Backfill done. processed={ok}, failed={failed}")
    logger.info("=" * 60)

    if not args.no_deploy:
        try:
            from trigger_deploy import trigger_deploy
            trigger_deploy()
        except Exception as e:  # noqa: BLE001 — never fatal
            logger.warning(f"Deploy trigger skipped/failed: {e}")


if __name__ == "__main__":
    main()

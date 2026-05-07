"""
Scheduler Service for Daily Papers Tool
Chạy cron job vào giờ VN cố định (UTC+7) để crawl papers của ngày hôm trước
"""
import os
import sys
import logging
import schedule
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Time to run
TIME = "10:00"

# Load environment variables
load_dotenv(override=False)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from daily_papers_tool import run_daily_digest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/scheduler.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def get_previous_day_date():
    """
    Lấy ngày hôm trước để crawl papers.
    Xử lý đúng edge case khi sang tháng mới hoặc năm mới.
    
    Returns:
        str: Date string in YYYY-MM-DD format
    """
    # Lấy thời gian hiện tại theo giờ VN (UTC+7)
    now_vn = datetime.now()
    
    # Lùi lại 1 ngày - timedelta xử lý đúng tất cả edge cases
    # bao gồm sang tháng mới, năm mới, leap year
    previous_day = now_vn - timedelta(days=1)
    
    date_str = previous_day.strftime('%Y-%m-%d')
    logger.info(f"Crawling papers for date: {date_str} (previous day from {now_vn.strftime('%Y-%m-%d')})")
    
    return date_str


def run_scheduled_job(model="moonshotai/kimi-k2.5", force_update=False):
    """
    Chạy job crawl papers cho ngày hôm trước
    """
    logger.info("=" * 60)
    logger.info("Starting scheduled daily paper digest job")
    logger.info("=" * 60)
    
    try:
        date_str = get_previous_day_date()
        logger.info(f"Running digest for date: {date_str} with model: {model}")
        
        result = run_daily_digest(date_str, model=model, force_update=force_update)
        
        if result:
            logger.info(f"Job completed successfully. Report saved to: {result}")
        else:
            logger.info(f"No papers found for {date_str}. Skipped this day.")
            
    except Exception as e:
        logger.error(f"Error in scheduled job: {e}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("Scheduled job finished")
    logger.info("=" * 60)


def main():
    f"""
    Main entry point cho scheduler service
    Schedule job chạy vào {TIME} trưa giờ VN mỗi ngày
    """
    # Get model from environment or use default
    model = os.getenv('LLM_MODEL', 'moonshotai/kimi-k2.5')
    
    logger.info("Starting Daily Papers Scheduler Service")
    logger.info(f"Using model: {model}")
    logger.info(f"Current time (VN): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Scheduled to run daily at {TIME} (noon) Vietnam time")
    
    # Schedule job at TIME Vietnam time
    schedule.every().day.at(TIME).do(run_scheduled_job, model=model)
    
    # Log next run time
    next_run = schedule.next_run()
    logger.info(f"Next scheduled run: {next_run}")
    
    # Run immediately on startup if specified
    if os.getenv('RUN_ON_STARTUP', 'false').lower() == 'true':
        logger.info("RUN_ON_STARTUP is enabled. Running job immediately...")
        run_scheduled_job(model=model)
    
    # Main loop
    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
            # Log heartbeat every hour
            current_minute = datetime.now().minute
            if current_minute == 0:
                logger.debug(f"Scheduler heartbeat. Next run: {schedule.next_run()}")
                
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Papers Scheduler Service")
    parser.add_argument(
        "--run-now", 
        action="store_true", 
        help="Run the job immediately (for testing)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model ID for NVIDIA NIM (overrides env variable, default: from .env)"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to process (YYYY-MM-DD). Only used with --run-now"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of papers to process (default: 0 = all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-summarize all papers and overwrite existing DB entries"
    )
    
    args = parser.parse_args()
    
    # Override model if specified
    if args.model:
        os.environ['LLM_MODEL'] = args.model
    
    if args.run_now:
        # Run once immediately (for testing)
        logger.info("Running in test mode (run once)")
        if args.date:
            logger.info(f"Processing specific date: {args.date}")
            run_daily_digest(args.date, model=os.getenv('LLM_MODEL', 'moonshotai/kimi-k2.5'), paper_limit=args.limit, force_update=args.force)
        else:
            run_scheduled_job(model=os.getenv('LLM_MODEL', 'moonshotai/kimi-k2.5'), force_update=args.force)
    else:
        # Run as scheduler service
        main()
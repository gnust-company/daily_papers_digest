"""
LangGraph nodes for the daily digest workflow.
"""
import os
import sys
import time
import logging
from datetime import datetime
# Add project root to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summary_utils.fetch_papers import fetch_daily_papers
from summary_utils.download_papers import download_all_papers
from summary_utils.summarize_papers import extract_text_from_pdf, summarize_paper, generate_markdown_from_summary
from summary_utils.upload_minio import upload_to_minio
from database.db_utils import (
    check_paper_exists,
    get_engine,
    save_paper_and_summary,
    get_papers_with_summaries,
)
from llm_provider import get_llm
from graph.state import DigestState

logger = logging.getLogger(__name__)


def fetch_papers_node(state: DigestState) -> dict:
    """Fetch papers from Hugging Face API for the given date."""
    date_str = state["date_str"]
    papers = fetch_daily_papers(date_str)
    limit = state.get("paper_limit", 0)
    if limit > 0:
        papers = papers[:limit]
        print(f"Limit enabled: processing first {limit} papers only")
    print(f"Fetched {len(papers)} papers for {date_str}")
    if not papers:
        print(f"Warning: No papers found for {date_str}")
    return {"papers": papers}


def download_pdfs_node(state: DigestState) -> dict:
    """Download PDFs for all fetched papers."""
    papers = state["papers"]
    date_str = state["date_str"]
    download_dir = os.path.join("papers", date_str)

    download_all_papers(papers, download_dir=download_dir)

    filtered_papers = []
    for paper in papers:
        local_path = paper.get("local_path")
        if local_path and os.path.exists(local_path):
            filtered_papers.append(paper)
        else:
            logger.warning(f"Paper {paper['id']}: PDF download failed or not found")

    logger.info(f"Downloaded {len(filtered_papers)} PDFs")
    return {"papers_to_process": filtered_papers}


def filter_duplicates_node(state: DigestState) -> dict:
    """Filter out papers that already have summaries in the database."""
    papers_to_process = state["papers_to_process"]

    # Force update mode: re-process all papers, skip duplicate check
    if state.get("force_update"):
        print("Force update mode: re-processing all papers.")
        return {"papers_to_process": papers_to_process}

    engine = get_engine()
    new_papers_only = []
    skipped_ids = set()

    for paper in papers_to_process:
        _paper_exists, summary_exists = check_paper_exists(paper["id"], engine)
        if summary_exists:
            print(f"Paper {paper['id']} already summarized. Skipping.")
            skipped_ids.add(paper["id"])
            continue
        # Keep if no summary (regardless of whether paper record exists)
        new_papers_only.append(paper)

    print(f"After duplicate check: {len(new_papers_only)} new papers to process")

    # Load existing summaries for skipped papers to include in report
    existing_summaries = []
    if skipped_ids:
        all_existing = get_papers_with_summaries(engine)
        for e in all_existing:
            paper_id = e.get("paper", {}).get("id")
            if paper_id in skipped_ids:
                summary_obj = e.get("summary")
                if summary_obj:
                    # Convert SQLAlchemy model to dict for report generation
                    existing_summaries.append({
                        "paper": e["paper"],
                        "summary": {
                            "tags": summary_obj.tags,
                            "main_problem": summary_obj.main_problem,
                            "main_idea": summary_obj.main_idea,
                            "main_results": summary_obj.main_results,
                            "conclusion_future_works": summary_obj.conclusion_future_works,
                            "publish_papers": summary_obj.publish_papers,
                            "patent_ideas": summary_obj.patent_ideas,
                        }
                    })
        print(f"Loaded {len(existing_summaries)} existing summaries for report")

    # Edge case: all papers are duplicates
    if not new_papers_only:
        print("All papers are duplicates. Generating report from existing data.")
        return {
            "papers_to_process": [],
            "summaries": existing_summaries,
        }

    return {
        "papers_to_process": new_papers_only,
        "summaries": existing_summaries,
    }


def extract_text_node(state: DigestState) -> dict:
    """Extract text from downloaded PDFs."""
    papers_to_process = state["papers_to_process"]
    updated_papers = []

    for paper in papers_to_process:
        local_path = paper.get("local_path")
        if not local_path or not os.path.exists(local_path):
            logger.warning(f"Paper {paper['id']}: local_path missing, skipping")
            continue

        text = extract_text_from_pdf(local_path)
        if not text or not text.strip():
            logger.warning(f"Paper {paper['id']}: empty text extraction, skipping")
            continue

        paper["_extracted_text"] = text
        paper["_text_length"] = len(text)
        updated_papers.append(paper)

    logger.info(f"Extracted text from {len(updated_papers)} PDFs")
    return {"papers_to_process": updated_papers}


def summarize_worker_node(state: dict) -> dict:
    """
    Worker node that summarizes a single paper.
    Receives a partial state with a single 'paper' dict.
    """
    paper = state.get("paper")
    if not paper:
        logger.error("summarize_worker_node received no paper in state")
        return {"failed_papers": []}

    text = paper.get("_extracted_text", "")
    if not text:
        logger.warning(f"Paper {paper['id']}: no extracted text, marking as failed")
        return {"failed_papers": [paper]}

    llm = get_llm()
    max_attempts = 3
    result = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = summarize_paper(paper, text, llm_instance=llm)
            if result:
                break
        except Exception as e:
            error_type = type(e).__name__
            logger.warning(
                f"Paper {paper['id']}: attempt {attempt}/{max_attempts} failed ({error_type}): {e}"
            )
            if attempt < max_attempts:
                wait_time = 2 ** (attempt - 1)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Paper {paper['id']}: all {max_attempts} attempts exhausted")

    if result:
        # Clean up heavy fields before storing to reduce memory
        paper.pop("_extracted_text", None)
        logger.info(f"Paper {paper['id']}: summarization succeeded")
        return {
            "summaries": [
                {
                    "paper": paper,
                    "summary": result,
                }
            ]
        }
    else:
        # Keep _extracted_text for potential retry
        logger.warning(f"Paper {paper['id']}: summarization failed")
        return {"failed_papers": [paper]}


def summarize_sequential_node(state: DigestState) -> dict:
    """
    Sequential summarization for all papers in papers_to_process.
    Used when MAX_PAPERS_PER_BATCH=1 to process papers one-by-one
    instead of dispatching parallel workers that would drop the rest.
    """
    papers_to_process = state.get("papers_to_process", [])
    if not papers_to_process:
        return {"summaries": [], "failed_papers": []}

    llm = get_llm()
    summaries = []
    failed_papers = []

    print(f"Processing {len(papers_to_process)} papers sequentially...")

    for paper in papers_to_process:
        paper_id = paper.get("id", "unknown")
        text = paper.get("_extracted_text", "")

        if not text:
            print(f"Paper {paper_id}: no extracted text, skipping")
            failed_papers.append(paper)
            continue

        result = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = summarize_paper(paper, text, llm_instance=llm)
                if result:
                    break
            except Exception as e:
                error_type = type(e).__name__
                print(f"Paper {paper_id}: attempt {attempt}/{max_attempts} failed ({error_type}): {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** (attempt - 1)
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Paper {paper_id}: all {max_attempts} attempts exhausted")

        if result:
            paper.pop("_extracted_text", None)
            summaries.append({"paper": paper, "summary": result})
            print(f"Paper {paper_id}: summarization succeeded")
        else:
            failed_papers.append(paper)
            print(f"Paper {paper_id}: summarization failed")

    print(f"Sequential processing complete: {len(summaries)} succeeded, {len(failed_papers)} failed")
    return {"summaries": summaries, "failed_papers": failed_papers}


def quality_gate_node(state: DigestState) -> dict:
    """Validate summaries and move failed ones to failed_papers."""
    summaries = state.get("summaries", [])
    failed_papers = state.get("failed_papers", [])
    valid_summaries = []
    newly_failed = []

    required_keys = [
        "tags",
        "main_problem",
        "main_idea",
        "main_results",
        "conclusion_future_works",
        "publish_papers",
        "patent_ideas",
    ]

    for entry in summaries:
        summary = entry.get("summary")
        paper = entry.get("paper", {})
        valid = True

        if not summary or not isinstance(summary, dict):
            valid = False
        else:
            # Check all required keys present
            for key in required_keys:
                if key not in summary:
                    logger.warning(f"Paper {paper.get('id')}: missing key '{key}' in summary")
                    valid = False
                    break

            # Validate main_problem length
            if valid and len(summary.get("main_problem", "")) < 20:
                logger.warning(f"Paper {paper.get('id')}: main_problem too short")
                valid = False

            # Validate tags non-empty
            if valid and not summary.get("tags"):
                logger.warning(f"Paper {paper.get('id')}: tags empty")
                valid = False

        if valid:
            valid_summaries.append(entry)
        else:
            newly_failed.append(paper)

    logger.info(
        f"Quality gate: {len(valid_summaries)} passed, {len(newly_failed)} failed"
    )
    return {
        "summaries": valid_summaries,
        "failed_papers": failed_papers + newly_failed,
    }


def save_to_db_node(state: DigestState) -> dict:
    """Save valid summaries to the database."""
    summaries = state.get("summaries", [])
    model_name = state.get("model_name", os.getenv("LLM_MODEL", "google/gemma-4-31b-it"))
    engine = get_engine()
    saved_count = 0

    for entry in summaries:
        paper = entry.get("paper", {})
        summary_json = entry.get("summary", {})
        text_length = paper.get("_text_length", 0)

        try:
            save_paper_and_summary(paper, summary_json, model_name, text_length, engine)
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save summary for {paper.get('id')}: {e}")
            continue
        finally:
            # Clean up private field after use
            paper.pop("_text_length", None)

    logger.info(f"Saved {saved_count} summaries to database")
    return {}


def generate_report_node(state: DigestState) -> dict:
    """Generate the markdown digest report."""
    date_str = state["date_str"]
    summaries = state.get("summaries", [])
    model_name = state.get("model_name", os.getenv("LLM_MODEL", "google/gemma-4-31b-it"))

    report = f"# 🤗 Daily Hugging Face Paper Digest - {date_str}\n\n"
    report += f"Báo cáo được tạo tự động vào lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} bằng mô hình `{model_name}`.\n\n"
    report += "## 📰 Summary of Papers\n\n"

    for i, entry in enumerate(summaries, 1):
        paper = entry.get("paper", {})
        summary = entry.get("summary", {})

        report += f"--- \n\n"
        report += f"## {i}. {paper.get('title', 'Untitled')} *({paper.get('upvotes', 0)} votes)*\n\n"
        # Show at most 5 authors; append "et al." when there are more.
        authors = paper.get('authors', [])
        if len(authors) > 5:
            authors_display = ', '.join(authors[:5]) + ', et al.'
        else:
            authors_display = ', '.join(authors)
        report += f"**Tác giả:** {authors_display}\n\n"

        published = paper.get("published_at")
        report += f"**Xuất bản lúc:** {published[:10] if published else 'N/A'}\n\n"

        if summary:
            report += generate_markdown_from_summary(summary, paper)
        else:
            report += "No summary available.\n\n"

        links = [
            ("Hugging Face", paper.get("hf_url", "N/A")),
            ("ArXiv Abstract", paper.get("arxiv_url", "N/A")),
            ("PDF Download", paper.get("pdf_url", "N/A")),
            ("Github Repository", paper.get("github_url")),
        ]

        report += "### Các đường dẫn liên quan\n\n"
        report += "| Nền tảng | Đường dẫn |\n"
        report += "| :--- | :--- |\n"
        for platform, link in links:
            if link and str(link).startswith("http"):
                report += f"| {platform} | [{link}]({link}) |\n"
            else:
                report += f"| {platform} | {link} |\n"
        report += "\n"

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    organized_dir = os.path.join("summaries", year, month)
    os.makedirs(organized_dir, exist_ok=True)

    report_path = os.path.join(organized_dir, f"daily_digest_{date_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report generated: {report_path}")
    return {"final_report": report, "report_path": report_path}


def upload_minio_node(state: DigestState) -> dict:
    """Upload the generated report to MinIO."""
    report_path = state.get("report_path")
    date_str = state["date_str"]

    if not report_path or not os.path.exists(report_path):
        logger.warning("Report path missing or file does not exist. Skipping MinIO upload.")
        return {}

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    object_name = f"summaries/{year}/{month}/daily_digest_{date_str}.md"

    try:
        upload_to_minio(report_path, object_name)
        logger.info(f"Uploaded report to MinIO: {object_name}")
    except Exception as e:
        logger.warning(f"MinIO upload failed: {e}")

    return {}


def cleanup_node(state: DigestState) -> dict:
    """Clean up temporary PDF files."""
    papers_to_process = state.get("papers_to_process", [])
    deleted_count = 0
    dirs_to_check = set()

    for paper in papers_to_process:
        local_path = paper.get("local_path")
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                deleted_count += 1
                dirs_to_check.add(os.path.dirname(local_path))
            except Exception as e:
                print(f"Failed to delete {local_path}: {e}")

    # Remove empty parent directories
    for directory in dirs_to_check:
        try:
            if os.path.isdir(directory) and not os.listdir(directory):
                os.rmdir(directory)
                print(f"Removed empty directory: {directory}")
        except Exception as e:
            print(f"Failed to remove directory {directory}: {e}")

    print(f"Cleaned up {deleted_count} temporary PDF files")
    return {}


def retry_failed_node(state: DigestState) -> dict:
    """Retry summarization for failed papers with higher temperature."""
    failed_papers = state.get("failed_papers", [])
    retry_count = state.get("retry_count", 0)
    new_summaries = []
    still_failed = []

    retry_count += 1
    logger.info(f"Retry attempt {retry_count}: re-processing {len(failed_papers)} failed papers")

    llm = get_llm(temperature=0.9)

    for paper in failed_papers:
        text = paper.get("_extracted_text", "")
        # If text was cleaned up, try to re-extract from file
        if not text:
            local_path = paper.get("local_path")
            if local_path and os.path.exists(local_path):
                text = extract_text_from_pdf(local_path)
                paper["_extracted_text"] = text
                paper["_text_length"] = len(text)
        if not text:
            still_failed.append(paper)
            continue

        try:
            result = summarize_paper(paper, text, llm_instance=llm)
            if result:
                new_summaries.append({"paper": paper, "summary": result})
            else:
                still_failed.append(paper)
        except Exception as e:
            logger.error(f"Retry failed for {paper.get('id')}: {e}")
            still_failed.append(paper)

    logger.info(
        f"Retry attempt {retry_count}: {len(new_summaries)} succeeded, {len(still_failed)} still failing"
    )

    return {
        "summaries": new_summaries,
        "failed_papers": still_failed,
        "retry_count": retry_count,
    }

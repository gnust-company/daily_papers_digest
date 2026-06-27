import os
import sys
import re
import pypdf
import json
import logging
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# Control chars 0x00-0x1F except normal whitespace (0x20)
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def _sanitize_text(text: str) -> str:
    """Remove control characters that corrupt LaTeX (e.g. \\r in \\rightarrow)."""
    if not isinstance(text, str):
        return text
    return _CONTROL_CHAR_RE.sub('', text).replace('\r\n', '\n').replace('\r', '')


def _sanitize_summary(summary: dict) -> dict:
    """Clean all string fields in a summary dict."""
    for key, value in summary.items():
        if isinstance(value, str):
            summary[key] = _sanitize_text(value)
        elif isinstance(value, list):
            summary[key] = [_sanitize_text(v) if isinstance(v, str) else v for v in value]
    return summary

# Add parent directory to path for importing llm_provider
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_provider import get_llm

# Define Pydantic models for structured output
class PaperSummary(BaseModel):
    """Structured summary of a research paper (concise restatement, NOT a translation)."""
    tags: List[str] = Field(
        description="3-8 short AI/ML keywords (e.g., RAG, Diffusion, GAN, LLMs)."
    )
    main_problem: str = Field(
        description="The core problem/gap this work tackles. Max 3 sentences, ~60 words."
    )
    main_idea: str = Field(
        description="The core approach/method proposed. Max 4 sentences, ~90 words."
    )
    main_results: str = Field(
        description="Key findings or metrics. Max 5 short bullet points, one line each."
    )
    conclusion_future_works: str = Field(
        description="Conclusion and future directions. Max 3 sentences, ~60 words."
    )
    publish_papers: List[str] = Field(
        description="Exactly 3 concise research-direction ideas, each 1-2 sentences."
    )
    patent_ideas: List[str] = Field(
        description="Exactly 3 concise practical/patent ideas (mobile-focused), each 1-2 sentences."
    )


# Per-paper raw-output dumps for debugging verbosity / parse failures.
# Each summarization attempt appends to logs/debug_summaries/<paper_id>.md.
DEBUG_SUMMARY_DIR = os.path.join("logs", "debug_summaries")


def _dump_summary_debug(paper_info, raw_result):
    """Append the model output for one summarization attempt to a debug md file.

    On a successful parse, writes the rendered summary (the readable result).
    On a parse failure (e.g. the model over-generated and hit the length limit,
    or returned an empty response), writes the raw model output + token usage so
    we can see exactly what the model produced instead of only the error.

    Never raises — debugging must not break the pipeline.
    """
    try:
        os.makedirs(DEBUG_SUMMARY_DIR, exist_ok=True)
        raw = raw_result.get("raw")
        parsed = raw_result.get("parsed")
        err = raw_result.get("parsing_error")

        if parsed is not None:
            label = "### Rendered summary (parsed OK)"
            body = generate_markdown_from_summary(parsed.model_dump(), paper_info)
        else:
            label = "### Raw model output (PARSE FAILED)"
            chunks = []
            if raw is not None:
                content = getattr(raw, "content", None)
                if content:
                    chunks.append(str(content))
                for tc in getattr(raw, "tool_calls", []) or []:
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                    if args is not None:
                        chunks.append(json.dumps(args, ensure_ascii=False, indent=2))
            body = "\n\n".join(chunks) if chunks else "(empty response)"

        meta = getattr(raw, "response_metadata", {}) if raw else {}
        usage = (meta.get("token_usage") or {}) if isinstance(meta, dict) else {}

        header = (
            f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — attempt\n"
            f"- paper: `{paper_info.get('id')}` — {paper_info.get('title', '')}\n"
            f"- status: {'PARSED OK' if parsed else 'PARSE FAILED'}\n"
            f"- tokens: prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"total={usage.get('total_tokens')}\n"
            f"- output_chars: {len(body)}\n"
        )
        if err:
            header += f"- parsing_error: {err}\n"
        header += f"\n{label}\n\n"

        path = os.path.join(DEBUG_SUMMARY_DIR, f"{paper_info.get('id')}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(header + body + "\n\n---\n\n")
    except Exception as e:  # noqa: BLE001 — debugging must never break the run
        logger.debug(f"debug dump failed for {paper_info.get('id')}: {e}")


def summarize_paper(paper_info, text, llm_instance=None):
    """
    Summarizes a paper using LLM based on the extracted text.
    Returns structured JSON data using LangChain with Pydantic model.
    
    Args:
        paper_info: Dictionary with paper metadata (id, title, etc.)
        text: Extracted text from PDF
        llm_instance: Pre-configured LLM instance (if None, creates one via get_llm())
    """
    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that summarizes research papers.

Your job is to DISTILL the paper into its essential points in concise Vietnamese —
NOT to translate it, NOT to paraphrase it at length, and NOT to reproduce the
source text. A reader must grasp what the paper does and why it matters in under
two minutes. Restate ideas in your own words; never copy or translate sentences
from the paper.

You must respond with structured data following the provided schema.

LENGTH BUDGET (hard limits — exceeding them wastes tokens and causes the output
to be truncated and discarded):
- tags: 3-8 short keywords.
- main_problem: max 3 sentences (~60 words). Just the gap/problem.
- main_idea: max 4 sentences (~90 words). Just the core approach/method.
- main_results: max 5 short bullet points, one line each (key findings/numbers).
- conclusion_future_works: max 3 sentences (~60 words).
- publish_papers: exactly 3 ideas, each 1-2 sentences.
- patent_ideas: exactly 3 ideas, each 1-2 sentences.
- Keep the TOTAL output under ~1,200 Vietnamese words (~2,500 tokens).

CONTENT RULES:
- Base everything on the extracted text only.
- Use Vietnamese for all fields, EXCEPT keep technical names in English (e.g.,
  "Vision-Language Action", not "Thị giác-Ngôn ngữ-Hành động").
- Tags = common AI/ML keywords (e.g., RAG, Diffusion, GAN, LLMs).
- Patent ideas: practical applications, especially mobile phones; explain without
  the paper's abbreviations.
- Each field value is Markdown to drop into the template; do NOT recreate the
  headings (### I. Main Problem, ### II. Main Idea, ...)."""),
        ("human", """Please summarize the following research paper based on the title and extracted text.

Title: {title}

Extracted text (first few pages):
{text}

My template for the summary is as follows:
```markdown
**Tag:**

### I. Main Problem:

### II. Main Idea:

### III. Main Results:

### IV. Conclusion & Future Works:

### V. Brainstorming Space:

#### 1. Publish Papers:

#### 2. Patent:
```

Please provide a structured summary following the schema.""")
    ])
    
    try:
        # Initialize LLM
        llm = llm_instance if llm_instance is not None else get_llm()

        # include_raw=True keeps the raw AIMessage even when the model
        # over-generates and JSON parsing fails, so we can inspect what it
        # actually produced (dumped to logs/debug_summaries/).
        structured_llm = llm.with_structured_output(PaperSummary, include_raw=True)

        # Create the chain
        chain = prompt_template | structured_llm

        # Clean text - remove problematic unicode characters
        clean_text = text[:50000].replace('\ud835', '')

        # Invoke the chain
        raw_result = chain.invoke({
            "title": paper_info['title'],
            "text": clean_text
        })

        # Always dump the attempt (success or failure) for inspection.
        _dump_summary_debug(paper_info, raw_result)

        parsed = raw_result.get("parsed")
        if parsed is None:
            logger.error(
                f"Error summarizing paper {paper_info['id']}: "
                f"{raw_result.get('parsing_error')}"
            )
            return None

        # Convert Pydantic model to dict for backward compatibility
        return _sanitize_summary(parsed.model_dump())

    except Exception as e:
        logger.error(f"Error summarizing paper {paper_info['id']}: {e}")
        return None


def extract_text_from_pdf(pdf_path, max_pages=10):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            num_pages = min(len(reader.pages), max_pages)
            for i in range(num_pages):
                text += reader.pages[i].extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
    return text


def generate_markdown_from_summary(summary_json, _paper_info):
    """
    Generates markdown content from structured JSON summary.
    This is used for backward compatibility with the report generation.
    """
    if not summary_json:
        return "Summary generation failed."

    # Sanitize in case summary came from DB with control chars
    summary_json = _sanitize_summary(summary_json)
    
    markdown = f"""
**Tag:** {', '.join(summary_json.get('tags', []))}

### I. Main Problem:
{summary_json.get('main_problem', 'N/A')}

### II. Main Idea:
{summary_json.get('main_idea', 'N/A')}

### III. Main Results:
{summary_json.get('main_results', 'N/A')}

### IV. Conclusion & Future Works:
{summary_json.get('conclusion_future_works', 'N/A')}

### V. Brainstorming Space:

#### 1. Publish Papers:
"""
    
    for i, idea in enumerate(summary_json.get('publish_papers', []), 1):
        markdown += f"{i}. {idea}\n"
    
    markdown += "\n#### 2. Patent:\n"
    for i, idea in enumerate(summary_json.get('patent_ideas', []), 1):
        markdown += f"{i}. {idea}\n"
    
    return markdown


if __name__ == "__main__":
    # Test with one of the downloaded papers
    sample_paper = {
        'id': '2512.23959',
        'title': 'Improving Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling',
        'hf_url': 'https://huggingface.co/papers/2512.23959',
        'arxiv_url': 'https://arxiv.org/abs/2512.23959'
    }
    pdf_path = "papers/2512.23959.pdf"
    if os.path.exists(pdf_path):
        print(f"Extracting text from {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        print("Summarizing...")
        summary_json = summarize_paper(sample_paper, text)
        print("\n--- Summary (JSON) ---\n")
        print(json.dumps(summary_json, indent=2, ensure_ascii=False))
        
        print("\n--- Summary (Markdown) ---\n")
        markdown = generate_markdown_from_summary(summary_json, sample_paper)
        print(markdown)

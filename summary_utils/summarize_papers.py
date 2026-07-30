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
from langchain_core.utils.json import parse_json_markdown

logger = logging.getLogger(__name__)

# Loaders used to rescue a JSON object the strict structured parser rejected,
# cheapest/safest first (see _salvage_summary). json_repair is optional: it adds
# recovery from trailing commas, single quotes and truncated JSON, but the
# pipeline works without it.
_JSON_LOADERS = [parse_json_markdown]
try:
    from json_repair import repair_json

    _JSON_LOADERS.append(lambda s: json.loads(repair_json(s)))
except ImportError:
    pass

# Control chars 0x00-0x1F except normal whitespace (0x20)
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

# The template emits a section title around each field ("### I. Main Problem",
# "#### 1. Publish Papers", ...). A field VALUE must never re-echo one of those
# titles — the template would then render it twice (the duplicated
# "### II. Main Idea" / "**Main Problem:**" the digests show). The model leaks
# the title in two shapes:
#   - a markdown heading at template level (1-3 '#'); '####'+ nests fine, so
#     it is allowed and the model may still use it to structure a value.
#   - a standalone bold label line, e.g. "**Main Problem:**".
# Both are stripped below. '-'/'1.' bullet lists are always allowed.
_BANNED_HEADING_RE = re.compile(r'^[ \t]{0,3}#{1,3}[ \t]')
_BOLD_LABEL_RE = re.compile(r'^[ \t]*\*\*[^*\n]{1,40}\*\*:?[ \t]*$')


def _is_leaked_label_line(line: str) -> bool:
    """A line that is a template-level heading OR a standalone bold section label."""
    return bool(_BANNED_HEADING_RE.match(line) or _BOLD_LABEL_RE.match(line))


def _field_has_leaked_label(value: str) -> bool:
    """True if `value` contains a leak worth spending a retry on.

    Mirrors what _normalize_field_headings actually changes, so we never burn a
    full ~50k-char retry on something we would keep anyway:
      - a banned heading (#/##/###) ANYWHERE -> it collides with the section
        title and gets demoted, so a retry is worth trying.
      - a bold label only when it LEADS the value (before any content) -> that
        is the duplicated-section-title bug.
    A bold label mid-value is legitimate structure ("**Hạn chế hiện tại:**"
    introducing a list), is kept by the normalizer, and must not trigger a retry.
    """
    if not isinstance(value, str):
        return False
    started = False
    for line in value.split('\n'):
        if _BANNED_HEADING_RE.match(line):
            return True
        if _BOLD_LABEL_RE.match(line):
            if not started:
                return True
            continue
        if line.strip():
            started = True
    return False


def _summary_has_leaked_label(summary: dict) -> bool:
    """True if any scalar field re-echoes a section title. Drives the retry."""
    return any(
        _field_has_leaked_label(v) for v in summary.values() if isinstance(v, str)
    )


def _normalize_field_headings(value: str) -> str:
    """Last-resort cleanup so a value is safe to drop under the template title.

    - A leaked label at the START of the value (before any real content) is
      dropped — that is the duplicated-section-title bug (heading or bold).
    - A heading leaked mid-value is DEMOTED to '####' so it nests below the
      section instead of colliding with it (preserves the content). A bold
      label leaked mid-value is left as-is (rare; may be legitimate emphasis).
    - '####'+ headings and bullet lists are always left untouched.
    """
    if not isinstance(value, str):
        return value
    out = []
    started = False  # have we passed the first non-blank content line?
    for line in value.split('\n'):
        if _BOLD_LABEL_RE.match(line):
            if not started:
                continue  # leading bold label -> drop
            started = True
            out.append(line)
        elif _BANNED_HEADING_RE.match(line):
            if not started:
                continue  # leading heading -> drop
            out.append(re.sub(r'^([ \t]{0,3})#{1,3}([ \t])', r'\1####\2', line))  # demote
            started = True
        else:
            if line.strip() != '':
                started = True
            out.append(line)
    return '\n'.join(out).lstrip('\n')


def _sanitize_text(text: str) -> str:
    """Remove control characters that corrupt LaTeX (e.g. \\r in \\rightarrow)."""
    if not isinstance(text, str):
        return text
    return _CONTROL_CHAR_RE.sub('', text).replace('\r\n', '\n').replace('\r', '')


def _sanitize_summary(summary: dict) -> dict:
    """Clean all string fields in a summary dict.

    Also normalizes template-level headings the model leaked into the scalar
    Markdown fields (see _normalize_field_headings). List fields (tags,
    publish_papers, patent_ideas) are item-level and never carry a heading, so
    they only get control-char cleaning.
    """
    for key, value in summary.items():
        if isinstance(value, str):
            summary[key] = _normalize_field_headings(_sanitize_text(value))
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
        description="The core problem/gap this work tackles."
    )
    main_idea: str = Field(
        description="The core approach/method proposed, ENDING with one everyday-life "
                    "analogy (ví dụ đời thường) that mirrors how the method actually "
                    "works, so a non-expert gets the intuition."
    )
    main_results: str = Field(
        description="Key findings or metrics, as short bullet points."
    )
    conclusion_future_works: str = Field(
        description="Conclusion and future directions."
    )
    publish_papers: List[str] = Field(
        description="Exactly 3 concise research-direction ideas."
    )
    patent_ideas: List[str] = Field(
        description="Exactly 3 concise practical/patent ideas (mobile-focused)."
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


def _raw_text_candidates(raw):
    """Every string in a raw AIMessage that might hold the JSON object.

    Depending on whether the model answered with a tool call or with plain
    content, the object lands in a different place — try both.
    """
    if raw is None:
        return []
    out = []
    content = getattr(raw, "content", None)
    if isinstance(content, str) and content.strip():
        out.append(content)
    elif isinstance(content, list):  # multi-part content blocks
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                out.append(block["text"])
    for tc in getattr(raw, "tool_calls", []) or []:
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if isinstance(args, str):
            out.append(args)
        elif isinstance(args, dict):
            out.append(json.dumps(args, ensure_ascii=False))
    return out


def _salvage_summary(raw):
    """Recover a PaperSummary from a raw response the structured parser rejected.

    Some models (e.g. hermes-agent) ignore "no code fence" and answer with
    ```json ... ``` — valid JSON wrapped in Markdown, which the strict parser
    rejects at column 1 and which used to drop the paper entirely. Two rescue
    layers, cheapest first (both are local — no extra LLM call):

    1. parse_json_markdown (langchain_core): strips the fence / surrounding
       prose and parses the object. Handles the fence case.
    2. json_repair (optional dependency): also fixes trailing commas, single
       quotes and truncated JSON. Skipped silently when not installed.

    Returns a PaperSummary, or None when nothing usable can be recovered.
    """
    for candidate in _raw_text_candidates(raw):
        for loader in _JSON_LOADERS:
            try:
                data = loader(candidate)
            except Exception:  # noqa: BLE001 — try the next loader/candidate
                continue
            if not isinstance(data, dict):
                continue
            try:
                return PaperSummary.model_validate(data)
            except Exception:  # noqa: BLE001 — parsed but wrong shape
                continue
    return None


# Models that turned out not to support with_structured_output (tool-calling).
# Keyed by model name so the wasted strict call is paid once per process, not
# once per paper — a strict call on a 50k-char prompt is expensive.
_NO_STRUCTURED_OUTPUT = set()


def _invoke_summary(llm, prompt_template, inputs):
    """Run one summarization call. Returns a dict like include_raw's result.

    The preferred path is with_structured_output (tool-calling): the server
    enforces the schema, so it is the strictest and cleanest.

    Some models served on NIM (e.g. hermes-agent) do not support it — they
    ignore the schema and answer with a ```json fence. The openai SDK then
    raises a ValidationError from inside its OWN parsing, before LangChain can
    hand back include_raw's `raw`, so the response is lost and the paper gets
    dropped. For those models we fall back to a plain call and parse the JSON
    out of the text ourselves; the prompt already carries a full JSON example,
    so the model still knows the exact shape.
    """
    model_name = getattr(llm, "model_name", None) or str(llm)

    if model_name not in _NO_STRUCTURED_OUTPUT:
        try:
            structured = llm.with_structured_output(PaperSummary, include_raw=True)
            return (prompt_template | structured).invoke(inputs)
        except Exception as e:  # noqa: BLE001 — any failure means: use plain mode
            _NO_STRUCTURED_OUTPUT.add(model_name)
            logger.warning(
                f"Model '{model_name}' does not support structured output "
                f"({type(e).__name__}); falling back to plain-call JSON parsing "
                f"for the rest of this run."
            )

    # Plain call: recover the object from the raw text.
    raw = (prompt_template | llm).invoke(inputs)
    parsed = _salvage_summary(raw)
    return {
        "raw": raw,
        "parsed": parsed,
        "parsing_error": None if parsed else "could not parse JSON from plain response",
    }


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

OUTPUT FORMAT (critical):
- Respond with ONLY the structured schema — no preamble, no reasoning, no
  conversational text, no markdown code fence. Some models (e.g. DeepSeek)
  tend to emit reasoning prose (like "We analyze...") before the JSON object;
  do NOT do that. The very first character of your response must be the
  opening brace of the structured object. Any leading text will cause the
  parser to reject the entire response and the paper will be silently dropped.
- Do not wrap the object in ```json fences. Output the raw object only.

LENGTH: There are NO per-field length caps — write each field as long as it
needs to be clear and useful. Stay concise (this is a digest, not a re-tell),
but do not pad. The TOTAL output must stay under ~2,500 tokens; exceeding that
truncates the JSON and the whole response is discarded.

CONTENT RULES:
- Base everything on the extracted text only.
- Use Vietnamese for all fields, EXCEPT keep technical names in English (e.g.,
  "Vision-Language Action", not "Thị giác-Ngôn ngữ-Hành động").
- Tags = common AI/ML keywords (e.g., RAG, Diffusion, GAN, LLMs).
- main_idea: describe the core approach FIRST, then END with one everyday-life
  analogy (ví dụ đời thường) of the method's MECHANISM (how it works), not of
  its topic — e.g. for an exploration-then-act robot, "giống như nhân viên mới
  bấm thử vài nút để học cách máy chạy trước khi vận hành". Concrete and vivid.
- Patent ideas: practical applications, especially mobile phones; explain without
  the paper's abbreviations.

MARKDOWN FORMAT — every field value IS Markdown, not plain text:
- Structure each field for SKIMMABILITY, not as one long prose wall. Use
  **bold** for key terms, "- " bullets for enumerations/steps/components, and
  short paragraphs to separate ideas.
- Pick the layout that fits the content: bullets when there is a list of
  distinct points; a short paragraph (with bold highlights) when prose reads
  better. There is no fixed shape — use what makes that field clearest.
- Use **bold** INLINE for emphasis only. Never write a standalone "**Label:**"
  line — those collide with the renderer and get stripped.
- The example below shows rich Markdown in action (main_problem, main_idea).

HEADING RULE (critical — the renderer adds section titles for you):
- Each field's value is ONLY that field's body content. The "### I. Main
  Problem", "### II. Main Idea", ... section titles are added automatically
  around your values — do NOT repeat them inside the value.
- Inside a value you MUST NOT use a level-1, level-2 or level-3 Markdown
  heading ("# ...", "## ...", "### ..."). Those collide with the section
  titles. If you want internal structure, use "####" or deeper, or bullet
  ("- ...") / numbered ("1. ...") lists instead."""),
        ("human", """Please summarize the following research paper based on the title and extracted text.

Title: {title}

Extracted text (first few pages):
{text}

Return your answer as the structured schema ONLY. The first character of your
response must be the opening brace — no prose, no "```json" fence, nothing else.
Each field's value is Markdown body — structure it for skimmability (bold key
terms, bullets), and never put "#", "##" or "### " headings inside it. Here is
a complete example of well-formed field values (content is illustrative only):

{{
  "tags": ["Vision-Language-Action", "Robotics", "In-Context Learning", "World Modeling"],
  "main_problem": "Các mô hình Vision-Language-Action hiện đại **thất bại khi triển khai trong thiết lập mới** (góc camera lạ, hình thái robot khác) vì chỉ điều kiện hóa trên quan sát hiện tại và chỉ dẫn ngôn ngữ, bỏ qua biến cấu hình hệ thống. Hậu quả:\\n- hiệu suất sụt giảm mạnh trong hoàn cảnh mới\\n- buộc **fine-tuning tốn kém** cho từng thiết lập riêng biệt.",
  "main_idea": "**In-Context World Modeling (ICWM)** định khung nhận diện hệ thống như bài toán thích ứng in-context. Quy trình:\\n- robot tự thực hiện chuỗi ngắn **động tác khám phá ngẫu nhiên**\\n- ghi lại các **chuyển tiếp trực quan**\\n- nối vào context window để Transformer **ngầm suy luận động lực học**\\n\\nKhác In-Context Learning truyền thống (dùng context để hiểu *hành vi*), ICWM dùng context để hiểu **hệ thống vận hành thế nào**, cho phép điều chỉnh chính sách mà không cập nhật tham số.\\n\\n**Ví dụ đời thường:** giống nhân viên mới thay vì đọc hướng dẫn, tự bấm thử vài nút và quan sát máy phản ứng để ngẩm hiểu cách nó chạy, rồi dùng được ngay.",
  "main_results": "- Trên LIBERO, ICWM vượt Multi-View BC +13.0% trên góc nhìn OOD.\\n- Tác vụ long-horizon hưởng lợi lớn nhất: +26.3% trên góc nhìn lạ.\\n- Robot UR5e thật: ICWM giữ hiệu suất cao khi chính sách chuẩn sụt từ 68% xuống 17%.\\n- Ablation: thiếu ảnh kết quả trong context làm sụt 56.4% hiệu suất.",
  "conclusion_future_works": "ICWM khắc phục điểm yếu khái quát hóa bằng cách chuyển cửa sổ context từ **định nghĩa hành vi** sang **nhận diện hệ thống**, cho phép tự hiệu chỉnh tại test-time không cần cập nhật tham số. **Hướng tương lai:** tối ưu chiến lược thăm dò chủ động và mở rộng cho môi trường động liên tục.",
  "publish_papers": [
    "Mở rộng nhận diện hệ thống in-context cho điều khiển đa robot, mỗi tác nhân dùng chuỗi tương tác tự sinh để đồng thời ước lượng động lực học bản thân và đối tác.",
    "Kết hợp world modeling ngầm với active learning để robot tự chọn động tác thăm dò thông tin nhất thay vì ngẫu nhiên.",
    "Áp dụng tư tưởng thăm dò-tương tác-suy luận cho xe tự hành để ngầm hiệu chỉnh mô hình động lực học trong điều kiện đường mới."
  ],
  "patent_ideas": [
    "Hệ thống tự hiệu chuẩn cánh tay robot công nghiệp: trước mỗi ca, robot thực hiện vài động tác thăm dò; camera ghi lại và mô hình suy luận tư thế camera + độ lệch động học để chỉnh chính sách ngay lần chạy đầu.",
    "Điều khiển robot gia đình qua điện thoại với camera tùy ý: người dùng đặt điện thoại bất kỳ; robot chuyển động thử ngắn, gửi video lên đám mây để phân tích tương quan không gian rồi thực hiện lệnh theo góc nhìn đó.",
    "Module nhận diện thay đổi phụ kiện cho robot lắp ráp: khi đổi đầu kẹp, robot chạy chương trình thăm dò ngắn để xác định độ dài tay và độ mở kẹp mới qua luồng hình ảnh, cập nhật tham số động lực học ẩn."
  ]
}}

Now produce the structured summary for the paper above. Remember: your
response must start with the opening brace of the structured object — no
preamble, no code fence, no reasoning.""")
    ])

    try:
        # Initialize LLM
        llm = llm_instance if llm_instance is not None else get_llm()

        # Clean text - remove problematic unicode characters
        clean_text = text[:50000].replace('\ud835', '')

        inputs = {"title": paper_info['title'], "text": clean_text}

        # One full call, plus at most ONE retry when the model leaked a section
        # title (#/##/### heading or a "**Main Problem:**" bold label) into a
        # field. A retry is a full ~50k-char call, so we cap it at 1; if the
        # retry is still dirty we fall through to _sanitize_summary, which
        # normalizes the leaked labels (drop/demote) so the paper is never
        # dropped just for a stray label.
        parsed = None
        for attempt in range(2):  # 0 = first try, 1 = single retry
            raw_result = _invoke_summary(llm, prompt_template, inputs)

            # Per-attempt dumps are off by default — set SUMMARY_DEBUG=true to
            # inspect exactly what the model returned (useful when diagnosing
            # parse failures or leaks).
            if os.getenv("SUMMARY_DEBUG", "").lower() == "true":
                _dump_summary_debug(paper_info, raw_result)

            parsed = raw_result.get("parsed")
            if parsed is None:
                # Last chance: the structured path can still hand back an
                # unparsed response (e.g. a ```json fence that slipped past the
                # server-side schema). Try to recover it locally before
                # dropping the paper. In plain mode this was already attempted,
                # so it simply returns None again.
                parsed = _salvage_summary(raw_result.get("raw"))
                if parsed is None:
                    logger.error(
                        f"Error summarizing paper {paper_info['id']}: "
                        f"{raw_result.get('parsing_error')}"
                    )
                    return None
                logger.warning(
                    f"Paper {paper_info['id']}: strict parse failed; recovered "
                    f"the JSON from the raw response."
                )

            if not _summary_has_leaked_label(parsed.model_dump()):
                break  # clean output — accept it

            if attempt == 0:
                logger.warning(
                    f"Paper {paper_info['id']}: model leaked a section title into "
                    f"a field; retrying once."
                )
            else:
                logger.warning(
                    f"Paper {paper_info['id']}: section title still present after "
                    f"retry; normalizing and accepting."
                )

        # Convert Pydantic model to dict; _sanitize_summary normalizes any
        # remaining template-level headings so the rendered digest stays clean.
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

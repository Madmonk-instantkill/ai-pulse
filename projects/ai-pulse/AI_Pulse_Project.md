# AI Pulse — Personal AI Intelligence Digest

## 1. Project Overview & Problem Statement

**Project Name:** AI Pulse — Personal AI Intelligence Digest

**Problem I'm Solving:**  
The AI and LLM field moves too quickly. Hundreds of research papers, model releases, engineering updates, and newsletters appear every week, but most are too technical or time-consuming to read in full.

**What I Want to Build:**  
An autonomous AI research assistant that runs every two days and produces a concise, beginner-friendly PDF digest. The system will:

1. **Discover** the most impactful and relevant AI research papers across Hugging Face, Semantic Scholar, and arXiv.
2. **Rank** candidates using recency, citation velocity, Hugging Face engagement, author reputation, and cross-source validation.
3. **Select the top three papers** related to Generative AI, LLMs, fine-tuning, reasoning, agents, RAG, multimodal models, and deep learning.
4. **Collect one newsletter per run** from a rotating set of three newsletters: TLDR AI, Alpha Signal, and Import AI.
5. **Read and digest** the selected papers and newsletter using MapReduce summarization so long documents do not overwhelm the LLM context window.
6. **Rewrite the material** in clear, plain English for a smart beginner who wants to become an AI expert.
7. **Extract a glossary** of three to five important technical terms from every source document.
8. **Verify every claim** in the final report against the original papers and newsletter using a dedicated Source-Traceability Verifier Agent.
9. **Generate and archive** a dated four-to-five-page PDF report.
10. **Email the report automatically** using SMTP and Windows Task Scheduler.

The system is intended to become a personal AI knowledge archive over time. Each run will be versioned and date-stamped so that previous reports, glossaries, source metadata, and verification results remain available for later review.

---

## 2. Design Philosophy

- **Specialized Multi-Agent Roles:** Each agent has one clear responsibility instead of one monolithic agent trying to perform the entire workflow.
- **CrewAI Orchestration:** CrewAI manages agents, tasks, task dependencies, structured outputs, retries, and the final workflow state.
- **MapReduce for Long Documents:** Papers and newsletters are split into manageable chunks, summarized independently, and then merged into a coherent document-level understanding.
- **Evidence Before Eloquence:** The Writer Agent may simplify the language, but it may not introduce claims that cannot be traced to an original source.
- **Clarity Over Unnecessary Completeness:** The report should provide approximately 80% conceptual understanding without overwhelming the reader with academic jargon.
- **Compounding Glossary:** Every source produces three to five glossary terms. The terms are stored across runs so the reader gradually builds technical vocabulary.
- **Rate-Limit Awareness:** Expensive API calls, especially author h-index lookups, are limited to shortlisted papers and cached locally.
- **Fully Autonomous Cadence:** The system runs every 48 hours on Windows with no manual intervention after setup.
- **Fault Tolerance:** A temporary API failure should not destroy the entire run. The system should use retries, cached results, fallback adapters, and a run manifest that records partial failures.

---

## 3. Recruiter Perspective 🎯

### What Impresses Hiring Managers

- **Authentic Problem Solving:** The project solves a genuine information-overload problem instead of being a generic tutorial clone.
- **Multi-Agent Orchestration:** CrewAI demonstrates how complex AI workflows can be decomposed into specialized, testable roles.
- **MapReduce Over Long Contexts:** The design addresses context limits, token cost, information dilution, and long-document processing.
- **Hybrid Data Integration:** The system combines structured API metadata with unstructured PDFs and newsletter content.
- **Source-Grounded Generation:** The final report is not accepted merely because it sounds plausible. Every important claim must be linked to source evidence.
- **Operational Reliability:** Caching, retries, rate-limit protection, run manifests, date-stamped outputs, and scheduled execution make the system closer to a real production workflow.

### Critical Questions & Architecture Defenses

1. **"How do you evaluate accuracy and avoid hallucinations?"**  
   **Defense:** A dedicated Source-Traceability Verifier Agent re-reads every factual claim in the draft report and checks it against source excerpts, page numbers, sections, or newsletter URLs. Unsupported or exaggerated claims are rewritten or removed before PDF generation.

2. **"What happens when APIs throttle or scrapers break?"**  
   **Defense:** The system uses exponential backoff, local caching, time-windowed candidate storage, fallback extraction adapters, and a two-stage ranking process that minimizes expensive API calls.

3. **"Why use citations, h-index, and upvotes together?"**  
   **Defense:** Citations are useful but lagging. Citation velocity helps identify newer papers that are gaining attention quickly. Hugging Face engagement provides an early community signal, while the lead author's h-index provides an author-reputation signal. Cross-source overlap provides an additional validation signal.

4. **"Why not send the entire paper to one LLM prompt?"**  
   **Defense:** Long prompts increase cost, context pressure, and the chance that important details are diluted. MapReduce produces traceable chunk summaries first and only then creates a document-level synthesis.

5. **"How do you handle young researchers with low h-indexes?"**  
   **Defense:** A lead-author h-index above 30 is a prioritization signal, not an absolute exclusion rule. Otherwise, the system could discard genuinely important work from newer researchers or new labs.

---

## 4. Discovery Sources & Scoring Architecture

### Context: Papers With Code Sunset

Papers With Code is not used as a discovery source in this architecture. The project uses native arXiv queries together with Hugging Face Daily Papers and Semantic Scholar so that the data pipeline remains explicit and easier to maintain.

### Three-Source Parallel Collector Pipeline

The collection phase runs the three source collectors in parallel.

| Source | Target Batch | Native Signals |
| :--- | :---: | :--- |
| **Hugging Face Daily Papers** | 30 papers | Upvotes, GitHub stars when available, recency |
| **Semantic Scholar API** | 50 papers | Citation count, authors, publication date |
| **arXiv API** | 30 papers | Title, abstract, authors, recency, full-text PDF link |

**Expected raw candidates:** approximately 110 papers  
**Expected unique candidates after deduplication:** approximately 60–80 papers

### Search Keyword Focus & Filtering

#### Positive Keyword Clusters

1. Large Language Models and Foundation Models
2. Fine-tuning, LoRA, QLoRA, and post-training
3. RLHF, DPO, RLVR, and reinforcement learning for language models
4. LLM agents, reasoning, tool use, and planning
5. RAG, context compression, and long-context models
6. Multimodal LLMs, including vision-language and audio-language models
7. Efficient training, inference optimization, and model compression

#### Negative Exclusions

Discard or strongly down-rank papers that match the following categories:

- Survey or review papers, unless explicitly allowed for a special topic
- Opinion or viewpoint papers
- Education or student prompt-engineering papers
- Healthcare applications or clinical case studies
- Enterprise chatbot adoption reports
- Papers with no meaningful connection to the project's AI/LLM topics

---

## 5. End-to-End Data Pipeline

### Step 1: Collect Candidates in Parallel

The Scout Agent launches the Hugging Face, Semantic Scholar, and arXiv collection tasks concurrently.

```text
HF API              -> 30 papers: upvotes, recency, optional GitHub stars
Semantic Scholar    -> 50 papers: citations, authors, publication date
arXiv API           -> 30 papers: abstract, recency, PDF URL

Total               -> approximately 110 raw candidates
```

Each collector converts its response into a common `PaperRecord` schema so that downstream tasks do not need to understand provider-specific response formats.

### Step 2: Deduplicate Candidates

Papers are deduplicated using a normalized arXiv identifier whenever one is available.

```text
1. Normalize arXiv identifiers.
2. Match papers across all three sources by arXiv ID.
3. Merge metadata rather than keeping duplicate records.
4. Preserve a list of the sources in which each paper appeared.
5. Set source_count to the number of distinct sources.
6. Flag papers appearing in two or more sources.
```

Expected result: approximately 60–80 unique paper records.

If an arXiv ID is unavailable, the system can use a fallback match based on normalized title, first author, and publication year. Fallback matches should be flagged for review because title matching is less reliable than identifier matching.

### Step 3: Enrich Missing Fields

After deduplication, the system fills missing signals where possible.

#### Citation Enrichment for Hugging Face Papers

For a Hugging Face paper without citation data:

```text
GET /graph/v1/paper/ArXiv:{arxiv_id}
    ?fields=citationCount
```

The returned `citationCount` is merged into the paper record.

#### Upvote Enrichment for Semantic Scholar Papers

For a Semantic Scholar paper without Hugging Face engagement data:

```text
GET /api/papers/{arxiv_id}
```

If the paper is not present on Hugging Face, `upvotes` is set to `0` and the record is marked with `hf_match = false` rather than treated as an API failure.

#### Lead-Author h-Index Enrichment

The project prioritizes the **lead author's h-index**. The preferred lookup is:

```text
GET /graph/v1/author/{authorId}?fields=hIndex
```

The system stores the result as `lead_author_h_index`.

An h-index above 30 is used as a positive prioritization signal. It is not a hard filter because high-quality work can come from early-career researchers and new laboratories.

#### Rate-Limit Protection

A naive implementation would make 60–80 author API calls for every run. Instead, the system uses a two-stage enrichment strategy:

1. **Stage 1 — Provisional Ranking:** Rank all deduplicated papers using available upvotes, citation velocity, recency, and source overlap. No author h-index calls are required.
2. **Stage 2 — Shortlist Enrichment:** Select the top 8–10 provisional candidates and fetch the lead-author h-index only for those papers.
3. **Stage 3 — Final Ranking:** Recalculate the complete score for the enriched shortlist and select the final top three.
4. **Caching:** Store author results in `author_cache.json`, keyed by Semantic Scholar author ID. Reuse cached values in future runs and attach a timestamp to every cache entry.

Operationally, every paper supports the complete schema, but only the shortlist needs expensive h-index enrichment before final selection.

### Step 4: Normalize, Score, and Rank

All available signals are normalized to a `0–1` range before weighting. The final list is sorted in descending score order, and the top three papers are selected.

---

## 6. Scoring & Ranking Architecture

### Core Scoring Problems & Solutions

#### Problem #1: Scale Mismatch

Upvotes, citations, and h-index values have very different numeric ranges. If raw values are used, citations can dominate the score.

**Solution:** Normalize each signal within the current candidate batch before applying weights.

#### Problem #2: Cold-Start Conflict

A paper published three days ago cannot reasonably compete with a two-year-old paper using raw citation counts alone.

**Solution:** Use citation velocity:

```text
citation_velocity = citation_count / max(days_since_published, 1)
```

This rewards papers that are gaining attention quickly while still allowing older papers with sustained influence to rank well.

#### Problem #3: Author-Reputation Ambiguity

The first author may be an early-career researcher while the last author may be the senior principal investigator. This project explicitly uses the **lead author's h-index** as the primary author-reputation signal because the research contribution is usually led by the first author.

The last author's h-index can be stored as an optional secondary field in a future version.

#### Cross-Source Validation Bonus

A paper that appears in multiple independent discovery sources has stronger evidence of relevance. The system assigns a cross-source bonus:

```text
cross_source_bonus = 0.1 * (source_count - 1)
```

The bonus is capped so it cannot overwhelm the core relevance signals.

### Normalization and Scoring Implementation

```python
def safe_normalize(value, batch_max):
    if batch_max <= 0:
        return 0.0
    return min(value / batch_max, 1.0)


def calculate_score(
    paper,
    max_upvotes_in_batch,
    max_citation_velocity_in_batch,
    max_lead_h_index_in_batch,
    today,
):
    # Normalize the primary signals to 0-1.
    upvote_norm = safe_normalize(
        paper.upvotes,
        max_upvotes_in_batch,
    )

    citation_velocity_norm = safe_normalize(
        paper.citation_velocity,
        max_citation_velocity_in_batch,
    )

    lead_h_index_norm = safe_normalize(
        paper.lead_author_h_index,
        max_lead_h_index_in_batch,
    )

    # A paper published today receives 1.0; papers older than 30 days receive 0.
    days_old = max((today - paper.published_date).days, 0)
    recency_norm = max(0.0, 1.0 - (days_old / 30.0))

    # Cross-source overlap: 0.0, 0.1, or 0.2 for one, two, or three sources.
    cross_source_bonus = min(
        0.1 * max(paper.source_count - 1, 0),
        0.2,
    )

    # Explicit preference for lead authors with h-index > 30.
    # This is a preference, not an exclusion rule.
    lead_author_priority_bonus = 0.05 if paper.lead_author_h_index > 30 else 0.0

    score = (
        (upvote_norm * 0.25)
        + (citation_velocity_norm * 0.25)
        + (lead_h_index_norm * 0.15)
        + (recency_norm * 0.20)
        + (cross_source_bonus * 0.15)
        + lead_author_priority_bonus
    )

    return score
```

The exact weights should be configurable in a settings file so that the ranking strategy can be evaluated and tuned without modifying the pipeline code.

### Paper Record Schema

Each deduplicated paper should be represented by a common structured record similar to the following:

```json
{
  "paper_id": "arxiv:2501.12345",
  "arxiv_id": "2501.12345",
  "title": "Example paper title",
  "authors": [],
  "lead_author_id": "semantic-scholar-author-id",
  "lead_author_name": "Author Name",
  "published_date": "2025-01-20",
  "abstract": "...",
  "pdf_url": "https://arxiv.org/pdf/2501.12345",
  "sources": ["huggingface", "semantic_scholar", "arxiv"],
  "source_count": 3,
  "upvotes": 0,
  "citation_count": 0,
  "citation_velocity": 0.0,
  "lead_author_h_index": 0,
  "hf_match": false,
  "provisional_score": 0.0,
  "final_score": 0.0
}
```

---

## 7. Newsletter Collection Strategy

### Newsletter Pool

The system uses exactly three newsletters and rotates them across runs:

1. **TLDR AI** — Rapid updates about AI models, research, tools, and developer news.
2. **Alpha Signal** — AI research, repositories, and emerging technical trends.
3. **Import AI** — Technical analysis, AI capabilities, research, and policy developments.

### Rotation Rule

Only one newsletter is collected and summarized for each two-day run.

```python
NEWSLETTER_ROTATION = [
    "TLDR AI",
    "Alpha Signal",
    "Import AI",
]

newsletter_for_run = NEWSLETTER_ROTATION[run_number % len(NEWSLETTER_ROTATION)]
```

The rotation state is stored in `run_state.json` so that a failed run does not accidentally skip a newsletter. If the selected newsletter cannot be retrieved, the system should retry it and then use the next available newsletter only after recording the fallback in the run manifest.

The newsletter is collected by a dedicated **Newsletter Collector Agent**. This agent is responsible only for locating the latest eligible issue, extracting its text, preserving the source URL and publication date, and returning structured content to the digestion pipeline.

---

## 8. MapReduce Processing for Papers and Newsletters

Long research papers should not be inserted into one oversized LLM prompt. The system uses a MapReduce workflow to preserve important details while controlling context size.

### Map Phase

For each document:

1. Download the paper PDF or newsletter content.
2. Extract text while preserving page, section, paragraph, and URL metadata.
3. Split the document into token-bounded chunks with small overlap.
4. Send each chunk to a Map Agent or Map task.
5. Ask the model to extract:
   - Main ideas
   - Methods and architecture
   - Important results
   - Limitations
   - Definitions of difficult terms
   - Exact evidence locations
6. Store one structured summary per chunk.

Every map summary should include a source locator such as `page 4, Methods`, `page 8, Results`, or a newsletter paragraph and URL.

### Reduce Phase

For each document, the Reduce Agent receives the chunk summaries rather than the full raw document. It should:

1. Merge repeated ideas.
2. Resolve conflicts between chunks.
3. Preserve important numbers and conditions.
4. Separate reported results from the authors' speculation.
5. Produce a document-level explanation.
6. Produce a list of candidate factual claims.
7. Produce a source-evidence ledger linking each claim to one or more chunk summaries.

### Why MapReduce Is Important

MapReduce allows the system to:

- Process papers larger than the model's comfortable context window.
- Run chunk-level summaries in parallel.
- Reduce prompt size and token cost.
- Maintain source locations for later verification.
- Reuse chunk summaries without repeatedly sending the full document.
- Preserve memory across tasks through structured intermediate files rather than relying on an LLM conversation history.

The same MapReduce pattern is used for the selected newsletter when the issue is long enough to require chunking.

---

## 9. Glossary and Long-Term Learning System

Because the goal is to become an AI expert, glossary extraction is a required step rather than an optional formatting feature.

### Per-Document Glossary

For every selected paper and the newsletter, the system extracts three to five important terms. Each glossary entry should contain:

- The technical term
- A plain-English definition
- Why the term matters in this document
- A short example or analogy
- The source document and evidence location

### Running Glossary Archive

Each run produces a dated glossary file, for example:

```text
outputs/
└── 2026-09-06_run_001/
    ├── AI_Pulse_2026-09-06.pdf
    ├── glossary_2026-09-06.md
    ├── source_evidence.json
    └── run_manifest.json
```

A separate `running_glossary.md` can be updated after each successful run. Duplicate terms should be merged rather than copied repeatedly, while new examples and explanations can be appended over time.

The Glossary Agent must not invent definitions. Definitions should be based on the source document and may be checked by the Source-Traceability Verifier Agent.

---

## 10. Multi-Agent Team — CrewAI + Remote LLM

The system is implemented with CrewAI agents and tasks. The CrewAI Flow or application controller manages state, ordering, parallel work, retries, and final delivery.

### Agent Roles

1. **Scout Agent**  
   Collects candidate papers from Hugging Face, Semantic Scholar, and arXiv. Applies keyword filters and returns normalized paper records.

2. **Deduplication and Enrichment Agent**  
   Matches records by arXiv ID, merges source metadata, calculates `source_count`, fills missing citations and upvotes, and coordinates lead-author h-index lookups using the cache.

3. **Ranking Agent**  
   Performs provisional ranking, requests shortlist enrichment when necessary, applies the weighted scoring formula, and selects the final top three papers.

4. **Newsletter Collector Agent**  
   Selects the newsletter for the current run, retrieves the latest eligible issue, extracts its text, and records its URL, title, and publication date.

5. **Map Summarizer Agent**  
   Processes document chunks independently. It extracts facts, methods, results, limitations, glossary candidates, and source locations.

6. **Reduce / Reader-Digester Agent**  
   Combines chunk summaries into a coherent paper-level or newsletter-level understanding without losing evidence links.

7. **Writer Agent**  
   Converts the structured digests into beginner-friendly explanations. It explains what changed, why it matters, how the method works, important limitations, and who should care.

8. **Glossary Agent**  
   Produces three to five source-grounded glossary terms per document and updates the running glossary archive.

9. **Source-Traceability Verifier Agent — Dedicated Final Gate**  
   This agent has one primary job: verify the final report against the original source material. It re-reads every factual claim in the simplified summary and asks:

   > Is this claim directly traceable to the source paper or newsletter, or did the Writer Agent invent, exaggerate, or overgeneralize it?

   For every claim, it records one of the following statuses:

   - `SUPPORTED` — directly supported by source evidence.
   - `PARTIALLY_SUPPORTED` — the source supports part of the claim, but the wording is too broad.
   - `UNSUPPORTED` — no evidence was found.
   - `NEEDS_REVISION` — evidence exists, but the claim needs clearer attribution or qualification.

   Unsupported and exaggerated claims must be rewritten, qualified, or removed before the report can be rendered. The verifier returns a pass/fail decision and a structured verification report.

10. **Report and Delivery Service**  
    This may be a deterministic Python service rather than an LLM agent. It renders the approved content into a PDF, archives all run artifacts, and sends the PDF by SMTP.

### Recommended CrewAI Workflow

```text
Discovery Crew
    ├── Parallel source collection
    ├── Deduplication
    ├── Provisional ranking
    ├── Shortlist enrichment
    └── Final top-three selection

Content Crew
    ├── Paper PDF retrieval
    ├── Newsletter retrieval
    ├── Map chunk summarization
    ├── Reduce document digestion
    └── Glossary extraction

Editorial Crew
    ├── Beginner-friendly report writing
    ├── Claim and evidence ledger creation
    ├── Source-Traceability Verifier final gate
    └── Revision loop for failed claims

Delivery Service
    ├── PDF rendering
    ├── Date-stamped archival
    ├── Run manifest creation
    └── SMTP email delivery
```

The PDF renderer must not run until the Source-Traceability Verifier Agent returns an approved result.

---

## 11. Evidence Ledger and Verification Protocol

The Writer Agent should produce a claim ledger alongside the draft report. The Source-Traceability Verifier Agent uses this ledger to check every claim efficiently.

### Claim Ledger Schema

```json
{
  "claim_id": "paper_01_claim_003",
  "document_id": "arxiv:2501.12345",
  "claim_text": "The method reduces inference cost by ...",
  "source_locator": "page 8, Results section",
  "source_excerpt": "...",
  "status": "SUPPORTED",
  "verifier_note": "The claim matches the reported experiment and preserves the stated conditions."
}
```

### Verification Loop

```text
1. Writer Agent creates the draft report and claim ledger.
2. Source-Traceability Verifier checks every factual claim.
3. If all material claims are SUPPORTED, approve the report.
4. If claims are PARTIALLY_SUPPORTED or NEEDS_REVISION, send them back for revision.
5. If claims are UNSUPPORTED, remove or rewrite them.
6. Run verification again after revision.
7. Render the PDF only after the report passes the verification gate.
```

The final PDF should include concise source references, but the detailed evidence ledger should remain archived as machine-readable JSON for auditability.

---

## 12. Report Structure

The final report should be concise enough to read in approximately 15 minutes and should target four to five pages.

### Suggested Sections

1. **Header and Run Metadata**
   - AI Pulse title
   - Run date and run ID
   - Newsletter selected for this run
   - Number of candidate papers collected

2. **Executive Summary**
   - Three to five high-level takeaways
   - One short explanation of why the selected topics matter

3. **Top Three Research Papers**
   For each paper:
   - Title and authors
   - Plain-English explanation
   - What the researchers changed or introduced
   - Why it matters
   - Important result or evidence
   - Limitations and caveats
   - Score and selection signals
   - Source link

4. **Newsletter Brief**
   - Main themes from the selected newsletter
   - The most relevant connection to the selected research papers
   - Important claims with source references

5. **Glossary**
   - Three to five terms per source, deduplicated where appropriate
   - Plain-English explanations

6. **Source Notes**
   - Links to papers and newsletter
   - A short note that claims were checked by the Source-Traceability Verifier Agent

---

## 13. Versioning, Storage, and Memory

Every successful run is date-stamped and stored as a separate archive. The cadence is every two days, not weekly.

### Suggested Directory Structure

```text
outputs/
├── 2026-09-06_run_001/
│   ├── AI_Pulse_2026-09-06.pdf
│   ├── glossary_2026-09-06.md
│   ├── selected_papers.json
│   ├── newsletter.json
│   ├── source_evidence.json
│   ├── verification_report.json
│   └── run_manifest.json
├── 2026-09-08_run_002/
│   └── ...
└── running_glossary.md

cache/
├── author_cache.json
├── paper_metadata_cache.json
├── newsletter_cache.json
└── seen_papers.json
```

### Memory Files

- `seen_papers.json` tracks papers from the previous ten runs to reduce repetition.
- `author_cache.json` stores lead-author h-index lookups and timestamps.
- `paper_metadata_cache.json` stores reusable API metadata.
- `newsletter_cache.json` stores issue URLs, dates, and extracted content hashes.
- `running_glossary.md` stores accumulated vocabulary and examples.
- `run_manifest.json` records the exact configuration, source counts, selected documents, failures, model settings, and timestamps for each run.

Each output should include a version such as `AI_Pulse_2026-09-06_v001.pdf` if more than one attempt is made on the same date.

---

## 14. Delivery & Automation Schedule

- **Artifact:** Four-to-five-page PDF generated with ReportLab.
- **Delivery:** Automated email delivery through SMTP.
- **Cadence:** Every 48 hours, triggered by Windows Task Scheduler.
- **Newsletter cadence:** One newsletter per run, rotating through TLDR AI, Alpha Signal, and Import AI.
- **Paper cadence:** Three newly selected research papers per run whenever enough eligible candidates are available.
- **Deduplication memory:** `seen_papers.json` tracking papers from the previous ten runs.
- **Verification requirement:** The PDF is not generated unless the Source-Traceability Verifier Agent approves the report.
- **Failure behavior:** If a run fails, preserve the failed-run manifest and cached intermediate data. Retry the failed stage rather than starting from zero when possible.

### Windows Task Scheduler Command

The exact command depends on the project environment, but the scheduled job should execute the main pipeline entry point every two days:

```text
python -m ai_pulse.main --run-now
```

The application should write logs to a dated file and return a non-zero exit code when the verification gate or delivery step fails.

---

## 15. Success Criteria

The project is successful when it can repeatedly complete the following workflow without manual intervention:

1. Collect approximately 110 raw candidates from three APIs.
2. If any duplicay, Deduplicate them into approximately 60–80 unique papers.
3. Enrich missing metadata without repeatedly hitting API rate limits.
4. Rank papers using normalized signals, citation velocity, lead-author h-index, recency, and source overlap.
5. Select the top three papers.
6. Collect exactly one newsletter for the current two-day rotation slot.
7. Process papers and the newsletter using MapReduce.
8. Produce three-to-five glossary terms per source document.
9. Generate a beginner-friendly draft with a claim ledger.
10. Verify the draft against original source material using the dedicated Source-Traceability Verifier Agent.
11. Render and archive a dated four-to-five-page PDF.
12. Email the approved PDF automatically.
13. Preserve enough structured memory to improve future runs and avoid repeating the same papers.

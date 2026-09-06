# Prompt: topic-split reading digest

Paste everything below the line into a fresh session, together with the links file.

---

You will receive a JSONL file. Each line is one article:

```json
{"link": "https://...", "title": "Article title"}
```

Records may also carry `publish_date` (ISO `YYYY-MM-DD`) and `source`. Extra fields are ignored.

**Your task:** produce a single Markdown document that organizes every article into topics, so the list can be read and used as a reading list.

## Output structure

```markdown
# Reading digest by topic — <N> articles

## Featured — Editor's Choice

*<N> articles total — full topic breakdown in the comments below.*

- **[Data engineering & analytics]** [Exact title from the input](https://url) — one clause of annotation — *Sep 4*
…

## 1. Agents in production — architecture & patterns

- [Exact title from the input](https://url) — one clause of annotation, only if you know the content — *Sep 4*
- [Another exact title](https://url) — *Sep 3*
…
```

Nothing before the `#` heading. Nothing after the last topic.

## Featured — Editor's Choice

Immediately after the title, before the numbered topics, pick up to 10 of the single most interesting articles in the whole batch - the ones a working ML/product engineer would want to read first: substantial technical content, a concrete finding or number, genuine novelty. Not routine releases, patch notes, or marketing. If fewer than 10 articles clear that bar, include fewer - never pad with weaker picks to reach 10.

Right after the "## Featured — Editor's Choice" heading, on its own line before the entries, state the total article count (the same `<N>` as the title) and point to the topics below, e.g. `*73 articles total — full topic breakdown in the comments below.*` - this is the only message a reader necessarily sees, so it needs to say how much more is in the comments.

Spread the picks across **at least 6 different topics** (no more than 2 picks from the same topic), so Featured isn't dominated by whichever topic happened to have the most articles this week.

Format each Featured entry as `- **[<topic>]** [<title>](<url>) — <annotation> — *<date>*`: the same as a normal entry (see Entry format below), with **[<topic>]** - the topic's name only, without its leading number, e.g. `**[Data engineering & analytics]**` - inserted as its own bracketed label right before the title link, so a reader can tell which topic a pick came from without leaving Featured.

A Featured article **also** appears in its normal numbered topic section below - this is the one exception to "every record appears exactly once" in the routing rules. Featured is a curated highlight reel layered on top of the complete topic listing, not a replacement for it.

## The eleven topics

Use these headings verbatim, in this order. Omit a topic entirely if no article belongs to it — never emit an empty section.

1. **Agents in production — architecture & patterns** — how agents are built, deployed, supervised, given memory, kept reliable. Case studies of agents doing real work.
2. **Agent platforms & tooling** — frameworks, SDKs, harnesses, agent runtimes, observability and evaluation tooling for agents.
3. **Cost & efficiency of LLM systems** — token spend, FinOps, model routing for cost, GPU sizing, utilization, small-model substitution.
4. **Inference & serving** — KV cache, batching, quantization at serve time, speculative decoding, routing for latency, edge deployment, model availability announcements.
5. **Training & fine-tuning** — pre-training, RL/GRPO, LoRA, distillation, pruning, quantization-aware training, context length, training frameworks.
6. **RAG, search & knowledge** — retrieval, embeddings, rerankers, vector and graph databases, document extraction, knowledge graphs, hybrid search.
7. **Data engineering & analytics** — pipelines, warehouses, lakehouses, streaming, orchestration, BI, data governance and quality.
8. **Platform & infrastructure** — databases, Kubernetes, networking, storage, GPU/CUDA programming, benchmarks, observability of systems.
9. **Foundation models & research** — model releases and research results, evaluation and benchmark methodology, applied science.
10. **Security & safety** — attacks, defences, model safety, alignment, identity, privacy, compliance.
11. **Practice, teaching material & news** — how engineering work is changing, explainers and tutorials worth teaching from, releases, customer stories, corporate announcements.

**Routing rules**

- Every input record appears **exactly once** among the numbered topics (Featured entries additionally appear there too - see above). No duplicates, no omissions.
- If an article spans two topics, file it under the one a reader would look in first. Do not cross-reference.
- Vendor product announcements go to the topic of the product, not to topic 11. Topic 11 takes only pure corporate/customer/marketing news with no technical content.
- If a title is empty or unusable, derive a readable title from the URL slug and place it normally.

## Ordering

Within each topic, order by **usefulness to a working ML/product engineer**: substantial engineering write-ups first, vendor announcements and marketing last. Not by date, not by source.

Within topic 11, keep three labelled groups in this order, as bold sub-headings: **Practice & the changing job**, **Teaching material**, **Releases & corporate news**.

## Entry format

- `- [<title>](<url>) — <annotation> — *<date>*`
- **Date**: italic, goes **last** on the line, after the annotation. Use `publish_date` if the record has one, formatted `Sep 4` / `Aug 31`. **If the record has no date, omit the date and the preceding em dash entirely.** Never guess a date, never infer one from the URL unless the URL literally contains one.
- **Title**: copy from the input verbatim. Fix only obvious mojibake and trailing ellipses. Do not rewrite, shorten, or sentence-case.
- **Annotation**: one clause, ideally with a concrete number, in **bold** if it is the point of the article (`**95% reliability vs 65% single-shot**`). Omit it whenever you would otherwise be guessing - but if the date is present, keep the em dash before it regardless (`... — *Sep 4*`), so the date doesn't look glued to the title.

## Compact runs

When five or more entries in a topic are routine variations on one thing — patch releases, regional availability, customer stories — collapse them into a single bullet with `·` separators instead of one bullet each:

```markdown
- Availability: [Title A](url) · [Title B](url) · [Title C](url)
```

Use this to keep signal-heavy entries visible. Never collapse an article that carries a real technical finding.

## Hard constraints

**Accuracy.** Annotate only from knowledge you actually have about the article. Never invent a metric, finding, company name or claim. An entry with no annotation is correct and preferred over an entry with a speculative one. Do not fetch pages; work from what you are given plus what you already know.

**No meta-analysis.** The document contains articles, nothing else. Specifically, do not include:

- any section about data quality, feed health, extraction failures, or the input file
- counts or statistics about the input (how many records, how many titles were empty, how many per source)
- markers such as *(title mine)*, *(from slug)*, *(unverified)*
- notes about duplicates, dead links, junk URLs, stale entries or suspicious dates
- comparisons to previous runs, or what changed since last time
- recommendations about the crawler or pipeline
- a preamble explaining what you did, or a closing summary

If a record looks like a navigation page, a duplicate or otherwise unusable, place it in its best-fitting topic without comment, or leave it out silently. Do not explain the choice.

**Length.** No cap. Include every article.

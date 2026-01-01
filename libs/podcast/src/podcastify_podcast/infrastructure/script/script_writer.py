from __future__ import annotations

import json
import os
import re
from pathlib import Path

from podcastify_contracts.podcast_job import Citation, EpisodeSegment
from podcastify_podcast.application.ports import ScriptWriter
from podcastify_podcast.domain.models import Chunk
from podcastify_podcast.infrastructure.llm.ollama import OllamaTextGenerator
from podcastify_podcast.infrastructure.logging import get_logger
from podcastify_podcast.infrastructure.text.overlap import (
    sentence_split,
    snippet_overlap,
    token_set,
)

log = get_logger(__name__)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 4000)

def _load_prompt(rel_path: str) -> str:
    env_dir = os.getenv("PROMPTS_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.is_dir():
            candidate = env_path / rel_path
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")

    cwd_path = Path.cwd() / "prompts"
    if cwd_path.is_dir():
        candidate = cwd_path / rel_path
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "prompts" / rel_path
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")

    return ""

class OllamaScriptWriter(ScriptWriter):
    def __init__(self, *, generator: OllamaTextGenerator, section_pass: bool = False) -> None:
        self.generator = generator
        self.section_pass = section_pass

    def write_script(
        self,
        *,
        chunks: list[Chunk],
        minutes: float,
        language: str,
        style: str,
    ) -> list[EpisodeSegment]:
        system = _load_prompt("script_system.md")
        user = _load_prompt("script_user.md")
        domain_hint = self._domain_hint(style)

        context_blocks: list[str] = []
        for idx, chunk in enumerate(chunks[:10], start=1):
            unique_pages = sorted({c.page for c in chunk.citations if c.page})
            context_blocks.append(
                f"Context {idx} (pages {unique_pages or ['?']}):\\n{chunk.text}"
            )
        context = "\\n\\n".join(context_blocks)[:MAX_CONTEXT_CHARS]

        format_instructions = (
            "Return JSON with a top-level key 'segments' that is a list of objects with fields: title, speaker, text. "
            "Do not include Markdown headings in the JSON. Structure MUST be: Hook, 3-6 body segments, Recap, Takeaway."
        )

        section_instructions = """
- Hook: 80-140 words. Use a vivid, daily-life analogy to grab attention.
- Body segments: 3-6 segments, each 100-220 words. Explain mechanisms with concrete examples and cite claims.
- Recap: 80-150 words. Reuse the main analogy and summarize 2-3 key points with citations.
- Takeaway: 80-140 words. One actionable insight plus the memory hook/analogy repeated.
"""

        prompt = f"""{system}

{user}
Domain hints: {domain_hint}

Language: {language}
Style: {style}
Target minutes: {minutes}

{format_instructions}
{section_instructions}

REFERENCE CONTEXT:
{context}
"""

        try:
            raw = self.generator.generate(prompt=prompt)
        except Exception as e:
            log.error("script generation failed, retrying with trimmed context: %s", e)
            # Fallback: smaller context and softer instructions
            fallback_blocks: list[str] = []
            for idx, chunk in enumerate(chunks[:3], start=1):
                unique_pages = sorted({c.page for c in chunk.citations if c.page})
                fallback_blocks.append(f"Context {idx} (pages {unique_pages or ['?']}):\n{chunk.text[:600]}")
            fallback_ctx = "\n\n".join(fallback_blocks)[: MAX_CONTEXT_CHARS // 2]
            fallback_prompt = (
                f"{system}\n\n{user}\nDomain hints: {domain_hint}\n"
                f"Language: {language}\nStyle: {style}\nTarget minutes: {minutes}\n"
                "Structure: Hook, 3 body, Recap, Takeaway. Keep each 80-140 words.\n"
                "Use only the context. Return JSON segments."
                f"\n\nREFERENCE CONTEXT:\n{fallback_ctx}"
            )
            raw = self.generator.generate(prompt=fallback_prompt)
        segments = self._normalize_structure(self._parse_segments(raw), chunks)
        segments = self._attach_citations(segments, chunks)
        segments = self._fact_check_segments(segments, chunks)
        segments = self._repair_citations(segments, chunks)
        segments = self._retry_or_flag_segments(segments, chunks)
        segments = self._regenerate_low_evidence(segments, chunks, style)
        if self.section_pass and chunks:
            segments = self._polish_sections(segments, chunks, style)
        segments = self._clamp_lengths(segments)
        segments = self._enforce_paragraph_citations(segments, chunks)
        segments = self._paragraph_snippet_validation(segments, chunks)
        segments = self._inline_citation_enforcer(segments, chunks)
        segments = self._citation_audit_pass(segments, chunks, style)

        # Attach citations from the most relevant chunks (preserve traceability)
        if chunks:
            flattened_citations: list[Citation] = []
            for chunk in chunks:
                flattened_citations.extend(chunk.citations)

            for seg in segments:
                if not seg.citations:
                    seg.citations = flattened_citations[:5]

        return segments

    def _parse_segments(self, raw: str) -> list[EpisodeSegment]:
        try:
            data = json.loads(self._extract_json(raw))
            segs = data.get("segments", data)
            if not isinstance(segs, list):
                raise ValueError("segments field is not a list")
            parsed = []
            for item in segs:
                parsed.append(
                    EpisodeSegment(
                        title=item.get("title") or "Segment",
                        speaker=item.get("speaker") or "Host",
                        text=item.get("text") or "",
                        citations=[],
                    )
                )
            if parsed:
                return parsed
        except Exception:
            pass

        # Fallback: treat entire response as one segment
        return [
            EpisodeSegment(
                title="Episode",
                speaker="Host",
                text=raw.strip(),
                citations=[],
            )
        ]

    def _extract_json(self, raw: str) -> str:
        """Pull first JSON object/array from a possibly chatty response."""
        match = re.search(r"\\{.*\\}|\\[.*\\]", raw, re.S)
        if match:
            return match.group(0)
        return raw

    def _normalize_structure(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """Guarantee a structured episode: Hook, 3–6 body segments, Recap, Takeaway."""
        cleaned: list[EpisodeSegment] = []
        for seg in segments:
            if not seg.text.strip():
                continue
            cleaned.append(self._trim_segment(seg))

        if not cleaned:
            cleaned = self._fallback_segments(chunks)

        # Ensure Hook at start
        if not cleaned or not cleaned[0].title.lower().startswith("hook"):
            hook_text = cleaned[0].text if cleaned else (chunks[0].text if chunks else "Intro to the topic.")
            cleaned.insert(0, EpisodeSegment(title="Hook", speaker="Host", text=hook_text, citations=[]))

        # Ensure Recap and Takeaway at end
        tail_titles = [seg.title.lower() for seg in cleaned[-2:]]
        if not tail_titles or "recap" not in " ".join(tail_titles):
            recap_text = cleaned[-1].text if cleaned else (chunks[-1].text if chunks else "Recap of the key ideas.")
            cleaned.append(EpisodeSegment(title="Recap", speaker="Host", text=recap_text, citations=[]))
        if not tail_titles or "takeaway" not in " ".join(tail_titles):
            takeaway_text = chunks[-1].text if chunks else "Main takeaway for the listener."
            cleaned.append(EpisodeSegment(title="Takeaway", speaker="Host", text=takeaway_text, citations=[]))

        # Enforce 3–6 body segments between Hook and final two
        if len(cleaned) >= 3:
            body = cleaned[1:-2]  # exclude hook and last two (recap/takeaway)
        else:
            body = []
        if len(body) < 3:
            for idx, chunk in enumerate(chunks):
                if len(body) >= 3:
                    break
                body.append(EpisodeSegment(title=f"Segment {idx+1}", speaker="Host", text=chunk.text, citations=[]))
        if len(body) > 6:
            body = body[:6]
        cleaned = [cleaned[0], *body, *cleaned[-2:]]

        # Limit total segments to 10
        cleaned = cleaned[:10]

        return [self._trim_segment(seg) for seg in cleaned]

    def _trim_segment(self, seg: EpisodeSegment) -> EpisodeSegment:
        words = seg.text.split()
        max_words = 220 if "recap" not in seg.title.lower() and "takeaway" not in seg.title.lower() else 140
        min_words = 60
        if len(words) > max_words:
            return EpisodeSegment(
                title=seg.title,
                speaker=seg.speaker,
                text=" ".join(words[:max_words]),
                citations=seg.citations,
            )
        if len(words) < min_words:
            return EpisodeSegment(
                title=seg.title,
                speaker=seg.speaker,
                text=" ".join(words * (min_words // max(1, len(words)) + 1))[: max_words * 6],
                citations=seg.citations,
            )
        return seg

    def _fallback_segments(self, chunks: list[Chunk]) -> list[EpisodeSegment]:
        fallback_texts = [c.text for c in chunks[:4]] or ["This episode summarizes the research."]
        built: list[EpisodeSegment] = []
        built.append(EpisodeSegment(title="Hook", speaker="Host", text=fallback_texts[0], citations=[]))
        for idx, txt in enumerate(fallback_texts[1:-1], start=1):
            built.append(EpisodeSegment(title=f"Segment {idx}", speaker="Host", text=txt, citations=[]))
        built.append(EpisodeSegment(title="Recap", speaker="Host", text=fallback_texts[-1], citations=[]))
        built.append(EpisodeSegment(title="Takeaway", speaker="Host", text="Key lesson for listeners.", citations=[]))
        return built

    def _attach_citations(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """Attach citations by matching segment text to closest chunks."""
        if not chunks:
            return segments
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception:
            return segments

        corpus = [c.text for c in chunks]
        bm25 = BM25Okapi([c.split() for c in corpus])
        enriched: list[EpisodeSegment] = []
        for seg in segments:
            if seg.citations:
                enriched.append(seg)
                continue
            scores = bm25.get_scores(seg.text.split())
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
            citations: list[Citation] = []
            for idx, score in ranked:
                if score <= 0:
                    continue
                citations.extend(chunks[idx].citations[:1])
            enriched.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=seg.text,
                    citations=citations[:5],
                )
            )
        return enriched

    def _repair_citations(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """If a segment lacks citations, try to attach from best-matching chunk; otherwise, replace with evidence note."""
        if not chunks:
            return [
                EpisodeSegment(
                    title=s.title,
                    speaker=s.speaker,
                    text=f"{s.text}\n\n[Evidence note: content limited due to missing citations.]",
                    citations=s.citations,
                )
                for s in segments
            ]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception:
            return segments

        bm25 = BM25Okapi([c.text.split() for c in chunks])
        repaired: list[EpisodeSegment] = []
        for seg in segments:
            if seg.citations:
                repaired.append(seg)
                continue

            scores = bm25.get_scores(seg.text.split())
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_chunk = chunks[best_idx]
            if scores[best_idx] <= 0:
                repaired.append(
                    EpisodeSegment(
                        title=seg.title,
                        speaker=seg.speaker,
                        text="[Content removed due to missing supporting citations.]",
                        citations=[],
                    )
                )
                continue

            snippet = best_chunk.text[:400]
            repaired.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=f"{seg.text}\n\n[Source snippet: {snippet}]",
                    citations=best_chunk.citations[:3],
                )
            )
        return repaired

    def _retry_or_flag_segments(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """If segments still have no citations, rebuild them from best chunk snippets."""
        if not chunks:
            return segments
        rebuilt: list[EpisodeSegment] = []
        for seg in segments:
            if seg.citations:
                rebuilt.append(seg)
                continue

            best_chunk = max(chunks, key=lambda c: len(c.citations))
            new_text = (
                f"{seg.title}: {best_chunk.text[:600]} "
                "[Reconstructed from source due to missing citations in generated text.]"
            )
            rebuilt.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=new_text,
                    citations=best_chunk.citations[:5],
                )
            )
        return rebuilt

    def _regenerate_low_evidence(self, segments: list[EpisodeSegment], chunks: list[Chunk], style: str) -> list[EpisodeSegment]:
        """Second-pass regeneration for weakly supported segments using top chunks as context."""
        if not chunks or not any(c.citations for c in chunks):
            return segments

        regenerated: list[EpisodeSegment] = []
        domain_hint = self._domain_hint(style)
        top_chunks_sorted = sorted(chunks, key=lambda c: len(c.citations), reverse=True)

        for seg in segments:
            needs_regen = (not seg.citations) or ("Fact check" in seg.text) or ("Evidence note" in seg.text) or ("Content removed" in seg.text)
            if not needs_regen:
                regenerated.append(seg)
                continue

            ctx = "\n---\n".join(c.text for c in top_chunks_sorted[:3])
            prompt = (
                f"You are a Nobel-level professor rewriting a podcast segment titled '{seg.title}'. "
                f"Use only the provided context. Include one concise, interpretable formula if it aids clarity, "
                f"and immediately explain it in plain language. Keep 120-220 words.\n"
                f"Domain hints: {domain_hint}\n\n"
                f"CONTEXT:\n{ctx}\n\n"
                "Return plain text only."
            )
            try:
                rewritten = self.generator.generate(prompt=prompt).strip()
            except Exception:
                rewritten = seg.text

            citations: list[Citation] = []
            for c in top_chunks_sorted[:3]:
                citations.extend(c.citations)

            regenerated.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=rewritten,
                    citations=citations[:5],
                )
            )

        return regenerated

    def _domain_hint(self, style: str) -> str:
        s = style.lower()
        if "math" in s:
            return "Use intuitive metaphors plus equations (limits, derivatives, probability ratios), but explain every symbol simply."
        if "physics" in s or "science" in s:
            return "Favor conservation, rates, energy/force analogies, and simple models (exponential decay/growth)."
        if "how" in s or "practical" in s:
            return "Step-by-step with simple ratios/rules of thumb; keep formulas minimal and concrete."
        return "Keep it rigorous yet accessible; use formulas only when they clarify."

    def _polish_sections(self, segments: list[EpisodeSegment], chunks: list[Chunk], style: str) -> list[EpisodeSegment]:
        """Rewrite sections (hook/body/recap/takeaway) with targeted prompts and best chunk context."""
        if not segments:
            return segments

        domain_hint = self._domain_hint(style)
        polished: list[EpisodeSegment] = []
        for idx, seg in enumerate(segments):
            role = self._section_role(idx, segments)
            best_chunk = self._best_chunk_for_segment(seg, chunks)
            ctx = "\n---\n".join([c.text for c in chunks[:2]] + ([best_chunk.text] if best_chunk else []))

            if role == "hook":
                target_min, target_max = 80, 160
            elif role == "takeaway":
                target_min, target_max = 90, 170
            elif role == "recap":
                target_min, target_max = 100, 200
            else:
                target_min, target_max = 130, 220

            prompt = self._section_prompt(role=role, domain_hint=domain_hint, target_min=target_min, target_max=target_max, ctx=ctx, current=seg.text)
            try:
                rewritten = self.generator.generate(prompt=prompt).strip()
            except Exception:
                rewritten = seg.text

            citations = seg.citations or (best_chunk.citations if best_chunk else [])
            polished.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=rewritten,
                    citations=citations[:5],
                )
            )

        return polished

    def _section_role(self, idx: int, segments: list[EpisodeSegment]) -> str:
        if idx == 0:
            return "hook"
        if idx == len(segments) - 2:
            return "recap"
        if idx == len(segments) - 1:
            return "takeaway"
        return "body"

    def _best_chunk_for_segment(self, seg: EpisodeSegment, chunks: list[Chunk]) -> Chunk | None:
        if not chunks:
            return None
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception:
            return chunks[0]
        bm25 = BM25Okapi([c.text.split() for c in chunks])
        scores = bm25.get_scores(seg.text.split())
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return chunks[best_idx]

    def _clamp_lengths(self, segments: list[EpisodeSegment]) -> list[EpisodeSegment]:
        """Hard clamp lengths per section to keep pacing tight."""
        clamped: list[EpisodeSegment] = []
        for idx, seg in enumerate(segments):
            role = self._section_role(idx, segments)
            words = seg.text.split()
            if role == "hook":
                min_w, max_w = 70, 170
            elif role in {"recap", "takeaway"}:
                min_w, max_w = 90, 190
            else:
                min_w, max_w = 130, 230

            if len(words) > max_w:
                words = words[:max_w]
            if len(words) < min_w:
                # pad lightly by echoing key phrases to meet pacing without hallucinating new claims
                extra = words[: min(len(words), 20)]
                while len(words) < min_w and extra:
                    words += extra

            clamped.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=" ".join(words),
                    citations=seg.citations,
                )
            )
        return clamped

    def _enforce_paragraph_citations(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """Ensure each paragraph is backed by at least one citation."""
        if not chunks:
            return segments
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception:
            return segments

        bm25 = BM25Okapi([c.text.split() for c in chunks])
        enforced: list[EpisodeSegment] = []
        for seg in segments:
            paragraphs = [p.strip() for p in seg.text.split("\n\n") if p.strip()]
            required = max(1, len(paragraphs))
            citations = list(seg.citations)
            if len(citations) < required:
                scores = bm25.get_scores(seg.text.split())
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                for idx, score in ranked:
                    if score <= 0:
                        continue
                    for cit in chunks[idx].citations:
                        citations.append(cit)
                        if len(citations) >= required + 2:
                            break
                    if len(citations) >= required + 2:
                        break

            note = ""
            if len(citations) < required:
                note = "\n\n[Citation note: some paragraphs have limited evidence in the provided context.]"

            enforced.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=seg.text + note,
                    citations=citations[: max(required, 3)],
                )
            )
        return enforced

    def _paragraph_snippet_validation(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """Drop or flag paragraphs that do not meaningfully overlap any citation snippet."""
        if not chunks or not segments:
            return segments

        page_snippets: dict[int, list[str]] = {}
        page_citations: dict[int, list[Citation]] = {}
        for ch in chunks:
            for cit in ch.citations:
                if cit.page is None:
                    continue
                page_citations.setdefault(cit.page, []).append(cit)
                if cit.snippet:
                    page_snippets.setdefault(cit.page, []).append(cit.snippet)

        fallback_snippets = [snip for snippets in page_snippets.values() for snip in snippets][:12]

        validated: list[EpisodeSegment] = []
        for seg in segments:
            paragraphs = [p.strip() for p in seg.text.split("\n\n") if p.strip()]
            if not paragraphs:
                validated.append(seg)
                continue

            candidates: list[tuple[int | None, str]] = []
            seen: set[tuple[int | None, str]] = set()
            for cit in seg.citations:
                page = cit.page
                if cit.snippet:
                    key = (page, cit.snippet)
                    if key not in seen:
                        candidates.append(key)
                        seen.add(key)
                if page is not None and page in page_snippets:
                    for snip in page_snippets[page]:
                        key = (page, snip)
                        if key not in seen:
                            candidates.append(key)
                            seen.add(key)

            if not candidates:
                if not fallback_snippets:
                    validated.append(seg)
                    continue
                candidates = [(None, snip) for snip in fallback_snippets]

            kept: list[str] = []
            dropped = 0
            new_citations = list(seg.citations)

            for para in paragraphs:
                best_score = 0.0
                best_page: int | None = None
                for page, snip in candidates:
                    if not snip:
                        continue
                    score = snippet_overlap(para, snip)
                    if score > best_score:
                        best_score = score
                        best_page = page

                if best_score < 0.08:
                    dropped += 1
                    continue

                tag = ""
                if best_score < 0.15:
                    tag = f" [Snippet check: weak support (overlap={best_score:.2f})]"

                if best_page is not None and best_page in page_citations:
                    if not any(c.page == best_page for c in new_citations):
                        new_citations.append(page_citations[best_page][0])
                    already_has_page = re.search(rf"\[p\.?\s*{best_page}\]", para)
                    if "p." not in tag and not already_has_page:
                        tag = f"{tag} [p.{best_page}]".strip()

                kept.append(f"{para}{tag}")

            if dropped and kept:
                kept.append(f"[{dropped} paragraph(s) removed for lack of snippet support]")
            if not kept:
                kept.append("[Content removed due to weak snippet support.]")

            validated.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text="\n\n".join(kept),
                    citations=new_citations[: max(3, len(new_citations))],
                )
            )

        return validated

    def _citation_audit_pass(self, segments: list[EpisodeSegment], chunks: list[Chunk], style: str) -> list[EpisodeSegment]:
        """Second pass: force per-paragraph citations; drop paragraphs without clear support."""
        if not chunks or not segments:
            return segments
        domain_hint = self._domain_hint(style)
        top_ctx = "\n---\n".join(c.text for c in chunks[:3])
        audited: list[EpisodeSegment] = []
        for seg in segments:
            paragraphs = [p.strip() for p in seg.text.split("\n\n") if p.strip()]
            if not paragraphs:
                audited.append(seg)
                continue
            prompt = (
                "You are a citation enforcer. For each paragraph below, keep it only if it is supported by the provided context. "
                "Add an inline citation in the form [source p.#] using the best-matching chunk. If unsupported, drop the paragraph.\n"
                f"Domain hints: {domain_hint}\n"
                f"CONTEXT:\n{top_ctx}\n\n"
                "PARAGRAPHS:\n" + "\n\n".join(paragraphs)
            )
            try:
                rewritten = self.generator.generate(prompt=prompt).strip()
            except Exception:
                rewritten = seg.text

            cleaned_paragraphs = [p.strip() for p in rewritten.split("\n") if p.strip()]
            text = "\n\n".join(cleaned_paragraphs) if cleaned_paragraphs else "[Content removed due to missing support.]"
            audited.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=text,
                    citations=seg.citations,
                )
            )
        return audited

    def _inline_citation_enforcer(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """Ensure each paragraph carries an inline page citation; inject and validate against known pages/snippets."""
        if not chunks:
            return segments
        page_pool = [(c.page, c.snippet) for ch in chunks for c in ch.citations if c.page]
        if not page_pool:
            return segments

        enforced: list[EpisodeSegment] = []
        for seg in segments:
            paragraphs = [p for p in seg.text.split("\n\n") if p.strip()]
            new_paras: list[str] = []
            for para in paragraphs:
                if self._has_valid_inline_citation(para, page_pool):
                    new_paras.append(para)
                    continue
                best_page = page_pool[0][0]
                new_paras.append(f"{para} [p.{best_page}]")
            enforced.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text="\n\n".join(new_paras) if new_paras else seg.text,
                    citations=seg.citations,
                )
            )
        return enforced

    def _has_valid_inline_citation(self, para: str, page_pool: list[tuple[int, str | None]]) -> bool:
        match = re.findall(r"\[p\.?\s*(\d+)\]", para, flags=re.I)
        if not match:
            return False
        pages = {int(m) for m in match}
        valid_pages = {p for p, _ in page_pool}
        return bool(pages & valid_pages)

    def _section_prompt(self, *, role: str, domain_hint: str, target_min: int, target_max: int, ctx: str, current: str) -> str:
        role_note = {
            "hook": "Grab attention with a vivid analogy and one sharp factual hook.",
            "body": "Explain mechanisms, use numbers or formulas when clarifying, and keep claims cited.",
            "recap": "Summarize 2-3 key points and repeat the main analogy.",
            "takeaway": "Give one actionable insight and restate the memory hook.",
        }.get(role, "Explain clearly with citations.")

        domain_role_note = self._role_domain_prompt(role, domain_hint)

        formula_prompt = ""
        if "math" in domain_hint.lower() or "science" in domain_hint.lower():
            formula_prompt = (
                "Prefer interpretable formulas (Bayes: P(H|D)=P(D|H)P(H)/P(D); exponential: N(t)=N0·e^{rt}; "
                "rate: derivative as slope; accumulation: integral). Immediately paraphrase what the symbols mean."
            )

        return (
            f"You are a Nobel-level professor polishing the {role} of a podcast.\n"
            f"Rewrite the text in {target_min}-{target_max} words. {role_note} "
            f"Include at least one citation-worthy claim and a concise formula if it aids clarity, with plain explanation.\n"
            f"{formula_prompt}\n"
            f"{domain_role_note}\n"
            f"Domain hints: {domain_hint}\n"
            f"CONTEXT (use only this):\n{ctx}\n\n"
            f"CURRENT TEXT:\n{current}\n\nReturn plain text only."
        )

    def _role_domain_prompt(self, role: str, domain_hint: str) -> str:
        """Add domain-specific guidance per section."""
        hint = domain_hint.lower()
        if "math" in hint:
            if role == "hook":
                return "Use a counterintuitive math intuition (e.g., infinity, limits) and one simple equation to intrigue."
            if role == "body":
                return "Lean on derivatives/integrals for trends and probability ratios; always unpack symbols in plain words."
            if role == "recap":
                return "Restate the key formula and its intuition; keep it concrete."
        if "physics" in hint or "science" in hint:
            if role == "hook":
                return "Open with a vivid physical metaphor (gravity, light, energy) and a single concrete fact."
            if role == "body":
                return "Use conservation/rate laws or exponential decay/growth; tie each to a sensory analogy."
            if role == "recap":
                return "Repeat the main metaphor and one governing equation, plainly."
        if "how" in hint or "practical" in hint:
            if role == "hook":
                return "Pose a relatable problem and a punchy promise; avoid heavy math."
            if role == "body":
                return "Step-by-step, with simple ratios or rules-of-thumb; keep one number per step."
            if role == "takeaway":
                return "Deliver one actionable step and a quick heuristic."
        return "Keep it rigorous yet accessible; favor one memorable analogy and a concise, explained formula."

    def _fact_check_segments(self, segments: list[EpisodeSegment], chunks: list[Chunk]) -> list[EpisodeSegment]:
        """Drop sentences with no overlap; flag low-support ones."""
        if not chunks:
            return segments
        corpus = [c.text for c in chunks]
        checked: list[EpisodeSegment] = []
        for seg in segments:
            sentences = sentence_split(seg.text)
            filtered: list[str] = []
            for sent in sentences:
                overlap = self._overlap_score(sent, corpus)
                if overlap < 0.15:
                    # Drop sentences with almost no support
                    continue
                if overlap < 0.25:
                    filtered.append(f"{sent} [Fact check: low support in provided context]")
                else:
                    filtered.append(sent)
            if not filtered:
                filtered.append("[Content removed due to lack of supporting evidence.]")
            checked.append(
                EpisodeSegment(
                    title=seg.title,
                    speaker=seg.speaker,
                    text=" ".join(filtered),
                    citations=seg.citations,
                )
            )
        return checked

    def _overlap_score(self, sentence: str, corpus: list[str]) -> float:
        stoks = token_set(sentence)
        best = 0.0
        for chunk in corpus:
            ctoks = token_set(chunk)
            if not ctoks:
                continue
            inter = len(stoks & ctoks)
            union = len(stoks | ctoks)
            score = inter / union if union else 0.0
            best = max(best, score)
        return best

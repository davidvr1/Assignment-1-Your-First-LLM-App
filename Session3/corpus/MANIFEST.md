# Corpus manifest — Israeli supplementary health insurance (שב"ן)

Domain carried over from Assignment 1/2 (Kupat Cholim "כללית" supplementary health plans),
expanded to three insurers so the eval set can include real multi-hop and cross-document
questions. All documents except the original PDF are **synthetic**, written for this exercise —
they mirror the structure of real Israeli שב"ן takanonim but are not legally authoritative and
should not be used for actual claims. Numbers were deliberately made to **overlap and conflict**
across documents (see `comparison_faq.md`) rather than being one-fact-per-paragraph, per the
assignment's warning against a too-easy corpus.

| File | Format | Source | Notes |
|---|---|---|---|
| `kelalit_moshlam_takanon.pdf` | PDF | real document (from Session1/Session2 era) | 20 pages. Text extraction is **genuinely garbled** — the embedded font uses a non-standard Hebrew glyph mapping, so `pypdf`/most parsers pull back reversed, cp1255-mangled text. This is the corpus's "messy document" (assignment §3, Task 3 reading exercise). |
| `kelalit_moshlam_summary.md` | Markdown | derived (Session1 `insurance.md`) | Clean, human-written Hebrew summary of the same PDF's plan — lets you compare a clean parse of a section against the mangled PDF extraction of the same content. |
| `maccabi_zahav_takanon.md` | Markdown | synthetic | Maccabi's competing plan (code `MZ-2025`). Deliberately cross-references Kelalit's numbers for multi-hop questions (organ transplant cap comparison). |
| `meuhedet_adif_takanon.md` | Markdown | synthetic | Meuhedet's competing plan (code `MA-ADIF-24`). Different cap structure (flat ₪500 co-pay for surgery instead of %), used for negation/exception questions (e.g. social freezing coverage). |
| `comparison_faq.md` | Markdown | synthetic | Cross-document FAQ built from the three takanonim above — the deliberate multi-hop source (Task 2 requires 2 multi-hop questions spanning documents). |
| `claims_process_guide.txt` | Plain text | synthetic | Generic claims-filing process, insurer-agnostic. Explicitly does **not** contain caps, waiting periods, or coverage lists — good source for unanswerable questions ("what's the appeal deadline for Maccabi?" isn't in this doc, and this doc says so about itself). |

**Formats represented:** PDF, Markdown, plain text (3 formats — exceeds the ≥2 requirement).
**Document count:** 6 (within the 5–10 range).

## Why this corpus, not something else
- Genuine parsing difficulty: the PDF's Hebrew glyph mangling is a real "read the model card /
  read your chunks" trap, not a contrived one.
- Genuine overlap/conflict: three insurers describe similar-sounding benefits (organ transplant
  caps, cancellation notice periods, fertility treatment) with different numbers and different
  scope definitions, which is exactly the ambiguity real RAG systems have to resolve rather than
  the single-fact-per-chunk case the assignment warns against.
- Genuine gaps: `claims_process_guide.txt` supports honest unanswerable questions (insurer-specific
  numbers it explicitly disclaims) without needing invented off-topic documents.

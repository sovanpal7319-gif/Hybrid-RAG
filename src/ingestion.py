"""Document ingestion.

Parses PDFs from the data/ directory into page-level text records, then
splits each page's text into overlapping token-based chunks. Embeddings and
vector store logic are deliberately out of scope here — this module's only
job is to get clean, citable, chunk-sized text out of raw PDFs.
"""

from pathlib import Path

import tiktoken
from pypdf import PdfReader

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, DATA_DIR

# cl100k_base is the encoding used by the embedding/chat models we expect to
# feed these chunks to, so token counts here match what those models will
# actually see (word or character counts can be off by a large margin).
_ENCODING = tiktoken.get_encoding("cl100k_base")


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from a single PDF, one record per page.

    We deliberately extract page-by-page rather than joining the whole
    document into one blob: downstream citation needs to point a user back
    to the exact page a claim came from, and that mapping is lost forever
    once pages are concatenated.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Scanned/image-only pages have no text layer, so extract_text()
        # returns "" (or occasionally None depending on the page content
        # stream). Passing that downstream would create empty chunks that
        # waste embedding calls and add noise with no retrievable content,
        # so we drop them here instead of letting later stages deal with it.
        if not text:
            continue

        pages.append(
            {
                "text": text,
                "page_number": page_number,
                "source": pdf_path.name,
            }
        )

    return pages


def parse_all_pdfs(data_dir: Path = DATA_DIR) -> list[dict]:
    """Run parse_pdf() over every PDF in data_dir and combine the results.

    Sorted glob so ingestion order is deterministic — otherwise the sample
    printed by the __main__ block below could vary between runs on the same
    filesystem.
    """
    all_pages = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        all_pages.extend(parse_pdf(pdf_path))

    return all_pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split a single string into overlapping chunks, measured in tokens.

    We slide a fixed-size window over the token stream rather than cutting
    it into back-to-back non-overlapping blocks. Without overlap, a fact or
    sentence that happens to fall right on a chunk boundary gets split in
    two, and neither half alone may contain enough context to be retrieved
    or understood correctly. Overlap guarantees the last `overlap` tokens of
    one chunk reappear as the first tokens of the next, so anything near a
    boundary still shows up whole in at least one chunk.
    """
    tokens = _ENCODING.encode(text)
    chunks = []

    # Advancing by (chunk_size - overlap) instead of chunk_size is what
    # creates the overlap: each new window starts before the previous one
    # ended, re-including its last `overlap` tokens. A smaller step means
    # more overlap (more redundancy, more chunks); a step equal to
    # chunk_size would mean zero overlap.
    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunks.append(_ENCODING.decode(tokens[start:end]))

        # The final window may already reach the end of the token stream
        # before its turn comes up in the range() sequence. Stop as soon as
        # that happens instead of continuing to emit trailing windows that
        # would just be smaller and smaller re-slices of the same tail
        # already captured.
        if end >= len(tokens):
            break

    return chunks


def chunk_page(
    page: dict,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Split one parsed page (as produced by parse_pdf/parse_all_pdfs) into
    overlapping chunk dicts, each carrying the page's own metadata.

    Chunking runs per page rather than across the whole document: a page
    number is only a meaningful citation if every chunk stays within the
    page it came from, so pages are never merged together here.
    """
    text_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)

    return [
        {
            "text": chunk,
            "page_number": page["page_number"],
            "source": page["source"],
            "chunk_id": f"{page['source']}_{page['page_number']}_{index}",
        }
        for index, chunk in enumerate(text_chunks)
    ]


def chunk_all_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Run chunk_page() over every page and flatten the results into one list."""
    all_chunks = []

    for page in pages:
        all_chunks.extend(chunk_page(page, chunk_size=chunk_size, overlap=overlap))

    return all_chunks


if __name__ == "__main__":
    pages = parse_all_pdfs()

    print(f"Total pages parsed: {len(pages)}")

    if pages:
        # A short raw sample is enough to eyeball whether extraction is
        # producing real sentences or garbled text (a common pypdf failure
        # mode with unusual fonts/encodings) without dumping a whole page.
        source_page = 235
        sample = pages[source_page]["text"][:300]
        print(f"Sample from {pages[source_page]['source']} (page {pages[source_page]['page_number']}):")
        print(sample)

    chunks = chunk_all_pages(pages)

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        sample_chunk = chunks[0]
        print(
            f"Sample chunk {sample_chunk['chunk_id']} "
            f"(source={sample_chunk['source']}, page={sample_chunk['page_number']}):"
        )
        print(sample_chunk["text"][:300])

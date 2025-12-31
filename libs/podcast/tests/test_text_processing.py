from podcastify_podcast.infrastructure.text.normalize import clean_page_text, drop_repeated_headers_footers
from podcastify_podcast.infrastructure.text.chunking import SentenceChunker
from podcastify_podcast.domain.models import PageText


def test_clean_page_text_hyphenation_and_whitespace():
    raw = "Tech-\nniques  \n\n great\tstuff\u00ad."
    assert clean_page_text(raw) == "Techniques\n\ngreat stuff."


def test_drop_repeated_headers_and_footers():
    pages = [
        PageText(page_number=1, text="Research Report\nIntro page content\nPage 1", source="doc.pdf"),
        PageText(page_number=2, text="Research Report\nSecond page content\nPage 2", source="doc.pdf"),
    ]

    cleaned = drop_repeated_headers_footers(pages)
    assert cleaned[0].text == "Intro page content"
    assert cleaned[1].text == "Second page content"


def test_sentence_chunker_preserves_citations_and_pages():
    pages = [
        PageText(
            page_number=1,
            text="Sentence one. Sentence two is longer. Sentence three is here.",
            source="doc.pdf",
        )
    ]
    chunker = SentenceChunker(max_chars=50, min_chars=10)
    chunks = chunker.chunk(pages)

    assert len(chunks) == 2
    assert len(chunks[0].citations) == 2
    assert all(c.page == 1 for c in chunks[0].citations + chunks[1].citations)

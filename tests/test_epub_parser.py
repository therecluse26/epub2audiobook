"""EPUB parsing: Roman numerals the parser is sure about, and Project Gutenberg's header and licence."""
from bs4 import BeautifulSoup
from ebooklib import epub

from epub2audiobook.epub_parser import Block, _blocks_from_body, _drop_gutenberg_text, load_book


def blocks(html):
    return _blocks_from_body(BeautifulSoup(f"<html><body>{html}</body></html>", "lxml").body)


def marked(b):
    return [b.text[s:e] for s, e in b.numerals]


def test_tagged_numerals_are_marked_and_the_text_is_unchanged():
    bs = blocks('<section epub:type="chapter"><hgroup><h2 epub:type="z3998:ordinal z3998:roman">I</h2>'
                '<p epub:type="title">Peril from the Sky</p></hgroup>'
                '<p>King Renard <span epub:type="z3998:roman">IV</span> shook his head.</p></section>')
    assert [(b.kind, b.text) for b in bs] == [("heading", "I. Peril from the Sky"),
                                             ("para", "King Renard IV shook his head.")]
    assert marked(bs[0]) == ["I"]
    assert marked(bs[1]) == ["IV"]


def test_a_block_holding_only_a_numeral_is_marked():
    bs = blocks("<h3>The Affair of the Pink Pearl</h3><h3>I</h3><p>II.</p><p>He was bigger than I.</p>"
                "<p>MIX.</p><p>IV</p>")
    assert [b.text for b in bs] == ["The Affair of the Pink Pearl. I", "II.", "He was bigger than I.", "MIX."]
    assert [marked(b) for b in bs] == [["I"], ["II"], [], []]  # "IV" alone is dropped as a page number


def test_a_drop_cap_is_not_a_numeral():
    bs = blocks("<p>I</p><p>t was late.</p>")
    assert [(b.text, b.numerals) for b in bs] == [("It was late.", [])]


def test_gutenberg_header_markup_is_removed():
    bs = blocks('<header class="pg-boilerplate pgheader" id="pg-header">'
                '<h2 id="pg-header-heading">The Project Gutenberg eBook of X</h2>'
                '<p>Release date: April 1, 1996 [eBook #485]</p>'
                '<div id="pg-start-separator"><span>*** START OF THE PROJECT GUTENBERG EBOOK X ***</span></div>'
                '</header><h2>Chapter 1</h2><p>Text.</p>'
                '<footer class="pg-boilerplate pgheader" id="pg-footer"><div id="pg-end-separator">'
                '<span>*** END OF THE PROJECT GUTENBERG EBOOK X ***</span></div><div>Licence.</div></footer>')
    assert [b.text for b in bs] == ["Chapter 1", "Text."]


def test_older_gutenberg_files_are_cut_at_the_start_and_end_lines():
    bs = [Block("para", "The Project Gutenberg EBook of X"), Block("para", "*** START OF THIS PROJECT GUTENBERG EBOOK X ***"),
          Block("break"), Block("para", "Text."), Block("para", "End of the Project Gutenberg EBook of X"),
          Block("para", "Licence.")]
    kept, ended = _drop_gutenberg_text(bs)
    assert [b.text for b in kept] == ["Text."]
    assert ended
    assert _drop_gutenberg_text([Block("para", "Text.")]) == ([Block("para", "Text.")], False)


def test_sections_after_an_old_gutenberg_end_line_are_skipped(tmp_path):
    book = epub.EpubBook()
    book.set_identifier("t")
    book.set_title("T")
    book.set_language("en")
    pages = ["<p>The Project Gutenberg EBook of T</p><p>*** START OF THIS PROJECT GUTENBERG EBOOK T ***</p>"
             "<h1>Chapter 1</h1><p>It begins.</p>",
             "<h1>Chapter 2</h1><p>It ends.</p><p>*** END OF THIS PROJECT GUTENBERG EBOOK T ***</p>",
             "<h1>The Full Project Gutenberg License</h1><p>Licence text.</p>"]
    items = []
    for i, html in enumerate(pages):
        item = epub.EpubHtml(title=f"Page {i}", file_name=f"page{i}.xhtml", lang="en")
        item.content = f"<html><body>{html}</body></html>"
        book.add_item(item)
        items.append(item)
    book.toc, book.spine = items, items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "t.epub"
    epub.write_epub(str(path), book)

    chapters = load_book(path).chapters
    assert [b.text for b in chapters[0].blocks] == ["Chapter 1", "It begins."]
    assert [b.text for b in chapters[1].blocks] == ["Chapter 2", "It ends."]
    assert (chapters[2].kind, chapters[2].include_default) == ("license", False)

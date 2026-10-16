"""Text normalisation: the speech-friendly rewrites (ConvertOptions.speech_friendly) and what they must leave alone."""
import pytest

from epub2audiobook.textproc import normalize_text


def spoken(text, heading=False, lang="a", numerals=()):
    return normalize_text(text, lang, is_heading=heading, speech_friendly=True, numerals=numerals)


@pytest.mark.parametrize("text, heading, expected", [
    # markdown leftovers and symbols
    ("## IX. Testing", True, "9. Testing."),
    ("He was **very** angry and *quite* tired.", False, "He was very angry and quite tired."),
    ("She read __the__ letter.", False, "She read the letter."),
    ("See [the map](https://example.com/map) here.", False, "See the map here."),
    ("![a cover](cover.png) The story begins.", False, "The story begins."),
    ("Use `grep` to search.", False, "Use grep to search."),
    ("He was #1 in his class.", False, "He was number 1 in his class."),
    ("Section 3.2 # Overview", True, "Section 3.2 Overview."),
    ("the end of it. * * * * * * And then", False, "the end of it. ... And then."),
    ("The total ======== 375.05.", False, "The total 375.05."),
    # Roman numerals after a section word
    ("Part I", True, "Part 1."),
    ("Act I, Scene V", True, "Act 1, Scene 5."),
    ("chapter ix", True, "chapter 9."),
    ("CHAPTER I Peril from the Sky", True, "CHAPTER 1 Peril from the Sky."),
    ("Sect. V. Of Relations.", True, "Sect. 5. Of Relations."),
    ("Book I. Of the Understanding. Part I. Of Ideas", True, "Book 1. Of the Understanding. Part 1. Of Ideas."),
    ("the dead, in Book XI of the Odyssey.", False, "the dead, in Book 11 of the Odyssey."),
    # Bible references, numbered paragraphs, list labels
    ("In Luke xiii. 3, we read:", False, "In Luke 13:3, we read:"),
    ("In 1 John iv. 10, it is written.", False, "In 1 John 4:10, it is written."),
    ('"St. Luke XI. 9," she said.', False, '"St. Luke 11:9," she said.'),
    ("loved by My Father (xiv. 20, 21).", False, "loved by My Father (14:20, 21)."),
    ("II. The second objection is this.", False, "2. The second objection is this."),
    ("Who then is guilty? (i) The servants. (ii) The nurse.", False,
     "Who then is guilty? (1) The servants. (2) The nurse."),
])
def test_speech_friendly_rewrites(text, heading, expected):
    assert spoken(text, heading) == expected


@pytest.mark.parametrize("text, heading", [
    ("He was bigger than I.", False),
    ("Bigger Than I", True),
    ("The King and I", True),
    ("The Part I Played", True),
    ("For my part I must own, that it is so.", False),
    ("I. M. Pei", True),
    ("V. S. Naipaul", True),
    ("Part C", True),
    ("Appendix C", True),
    ("the letter X marked it.", False),
    ('and make Our abode with him" (v. 23).', False),
    ("See v. 23 above.", False),
    ("the case of Rex v. Bailey which", False),
    ("We have d(i)spatched him.", False),
    ("written in C# and F#.", False),
    ("f*ck that, 2*3*4 is my_var_name.", False),
    ("M. de Pouillac and D. H. Lawrence, M.D.", False),
    ("Louis XIV ruled France.", False),  # untagged regnal numbers are not handled
])
def test_speech_friendly_leaves_these_alone(text, heading):
    assert spoken(text, heading) == normalize_text(text, "a", is_heading=heading)


def test_marked_numerals_become_digits_only_with_the_option():
    text = "King Renard IV shook his head."
    assert spoken(text, numerals=[(12, 14)]) == "King Renard 4 shook his head."
    assert normalize_text(text, "a", numerals=[(12, 14)]) == text
    # the footnote rule deletes " (3)" after a full stop; a converted label must survive it
    assert spoken("guilty? (i) The servants.", numerals=[(9, 10)]) == "guilty? (1) The servants."


def test_without_the_option_numerals_stay():
    assert normalize_text("Part I", "a", is_heading=True) == "Part I."
    assert normalize_text("In Luke xiii. 3, we read:", "a") == "In Luke xiii. 3, we read:"


def test_section_words_and_number_sign_follow_the_language():
    assert spoken("Il capitolo IX e la Parte I.", lang="i") == "Il capitolo 9 e la Parte 1."
    assert spoken("le #3 du chapitre V", lang="f") == "le numéro 3 du chapitre 5."

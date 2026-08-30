from drzavni_posao.tiktok_eval.extractors import (
    Episode,
    EpisodeCatalog,
    extract_episode,
    extract_query,
)


def _catalog() -> EpisodeCatalog:
    return EpisodeCatalog(
        [
            Episode(23, "Šah", "video-23"),
            Episode(38, "Svadba", "video-38"),
            Episode(120, "Svadba", "video-120"),
            Episode(1694, "Živanov jubilarac", "video-1694"),
            Episode(2000, "Noć veštica", "video-2000"),
            Episode(253, "Grejanje", "video-253"),
            Episode(400, "Brat od strica", "video-400"),
        ]
    )


def test_extracts_strict_latin_episode_markers() -> None:
    catalog = _catalog()

    for caption in ("Ep. 1694", "Epizoda 1694", "Epizoda br. 1694"):
        result = extract_episode(caption, catalog)
        assert result.ok
        assert result.episode_number == 1694
        assert result.youtube_video_id == "video-1694"
        assert result.episode_id == "video-1694"
        assert result.method == "episode_number"


def test_extracts_cyrillic_episode_marker_and_transliterated_title() -> None:
    result = extract_episode("Епизода бр. 2000: Ноћ вештица", _catalog())

    assert result.ok
    assert result.episode_number == 2000
    assert result.method == "number_and_title"


def test_does_not_treat_unlabelled_numbers_or_dates_as_episode() -> None:
    result = extract_episode(
        "Objavljeno 24.09.2021, pogledalo 1694 ljudi, sezona 12",
        _catalog(),
    )

    assert result.status == "not_found"
    assert result.episode_number is None


def test_unknown_explicit_episode_number_is_not_rescued_by_title() -> None:
    result = extract_episode("Ep. 9999: Živanov jubilarac", _catalog())

    assert result.status == "not_in_catalog"
    assert result.episode_number == 9999
    assert result.youtube_video_id is None


def test_matching_number_and_title_increases_confidence() -> None:
    result = extract_episode(
        "Replying to @neko Ep.1694: Zivanov jubilarac #drzavniposao",
        _catalog(),
    )

    assert result.ok
    assert result.title == "Živanov jubilarac"
    assert result.confidence == 1.0
    assert result.method == "number_and_title"


def test_number_and_different_recognised_title_are_rejected() -> None:
    result = extract_episode("Ep.1694: Noć veštica", _catalog())

    assert result.status == "conflict"
    assert result.reason == "episode_number_title_mismatch"
    assert result.youtube_video_id is None


def test_expected_number_title_wins_over_an_unrelated_longer_title_phrase() -> None:
    result = extract_episode(
        "Replying to @ennci8 BRAT OD STRICA ... Epizoda 253 Grejanje",
        _catalog(),
    )

    assert result.ok
    assert result.episode_number == 253
    assert result.title == "Grejanje"
    assert result.method == "number_and_title"


def test_multiple_different_episode_markers_are_rejected() -> None:
    result = extract_episode("Ep. 23 ili Epizoda 1694", _catalog())

    assert result.status == "conflict"
    assert result.reason == "multiple_episode_numbers"


def test_unique_title_fallback_accepts_minor_typo() -> None:
    result = extract_episode("Replying to @gledalac Zivanov jubilaarac", _catalog())

    assert result.ok
    assert result.episode_number == 1694
    assert result.method == "title_fuzzy"


def test_duplicate_title_without_episode_number_is_ambiguous() -> None:
    result = extract_episode("Replying to @gledalac Svadba", _catalog())

    assert result.status == "ambiguous"
    assert result.reason == "title_maps_to_multiple_episodes"


def test_duplicate_title_is_disambiguated_by_matching_number() -> None:
    result = extract_episode("Epizoda 120: Svadba", _catalog())

    assert result.ok
    assert result.episode_number == 120
    assert result.method == "number_and_title"


def test_query_consensus_removes_reply_header_and_preserves_raw_style() -> None:
    result = extract_query(
        [
            "Reply to @mika's comment\nona epizoda u kojoj Boškić slavi?",
            "Replying to @mika\nona epizoda u kojoj Boškić slavi?",
            "ona epizoda u kojoj Boskic slavi?",
        ]
    )

    assert result.ok
    assert result.text == "ona epizoda u kojoj Boškić slavi?"
    assert "Reply" not in result.text
    assert result.confidence > 0.9
    assert result.method == "query_cue_consensus"


def test_query_consensus_removes_inline_reply_header() -> None:
    result = extract_query(
        [
            "Reply to @mika's comment ona epizoda u kojoj Boškić slavi?",
            "Replying to @mika ona epizoda u kojoj Boškić slavi?",
        ]
    )

    assert result.ok
    assert result.text == "ona epizoda u kojoj Boškić slavi?"


def test_query_consensus_tolerates_missing_at_sign_and_trailing_header_noise() -> None:
    query = "Koja je ona gde viče na Boškića?"
    result = extract_query(
        [
            f"Reply to ennci8’s comment @=* {query}",
            f"Reply to ennci8’s comment @=* {query}",
        ]
    )

    assert result.ok
    assert result.text == query


def test_query_consensus_removes_ocr_noise_before_reply_header() -> None:
    query = "Kako je Živan postao alkos?"
    result = extract_query(
        [
            f"4 — be F _ Reply to kekcina’s comment ) {query}",
            f"icons Reply to kekcina’s comment ) {query}",
        ]
    )

    assert result.ok
    assert result.text == query


def test_query_consensus_removes_tiktok_sticker_ui_marker() -> None:
    result = extract_query(
        [
            "epizoda kad su išli da kose travu [Sticker]",
            "epizoda kad su išli da kose travu [Sticker]",
        ]
    )

    assert result.ok
    assert result.text == "epizoda kad su išli da kose travu"


def test_query_consensus_supports_cyrillic_without_transliterating_output() -> None:
    result = extract_query(
        [
            "Одговор на коментар @пера\nона епизода у којој Бошкић слави?",
            "она епизода у којој Бошкић слави?",
        ]
    )

    assert result.ok
    assert result.text == "она епизода у којој Бошкић слави?"


def test_single_good_query_frame_is_low_confidence() -> None:
    result = extract_query(["ona epizoda u kojoj Boškić slavi?"])

    assert result.status == "low_confidence"
    assert result.text == "ona epizoda u kojoj Boškić slavi?"
    assert result.reason == "consensus_requires_multiple_frames"


def test_short_or_ui_only_ocr_is_not_a_query() -> None:
    result = extract_query(["Replying to @mika", "Ep. 23", "super 😂"])

    assert result.status == "not_found"
    assert result.text is None


def test_disagreeing_frames_do_not_produce_an_accepted_query() -> None:
    result = extract_query(
        [
            "ona epizoda u kojoj Boškić slavi rođendan",
            "kada Čvarkov priča o sasvim drugoj stvari",
            "koja je epizoda sa Draganom na železnici",
        ]
    )

    assert result.status == "low_confidence"
    assert result.confidence < 0.78


def test_stable_query_is_isolated_from_varying_full_frame_ocr_noise() -> None:
    query = (
        "boškić nešto slavi, ovi mu uništili sve i čvarkov imitira šilovićku "
        "kako diriguje i njeni patuljci pevaju ‘guba gaba guba gaba’?"
    )
    result = extract_query(
        [
            f"10:31 LIVE For You {query} Share Follow",
            f"TikTok 82.4K views creator_name {query} Add comment...",
            f"random navigation labels {query} 12 345 Home Friends Inbox",
            f"Reply to @gledalac's comment\nnoise from subtitles\n{query}\n❤️ 831",
            f"AUTO CAPTIONS unrelated dialogue {query} Search drzavni posao",
        ]
    )

    assert result.ok
    assert result.text == query
    assert result.method == "stable_substring_consensus"
    assert result.confidence == 1.0


def test_query_cue_consensus_keeps_full_question_across_realistic_ocr_noise() -> None:
    result = extract_query(
        [
            (
                ". — random map labels e kako se zove ss he epizoda kada je "
                "boSkić nešto slavio, ovi mu uništili sve i Čvarkov imitira "
                "Šilovićku kako diriguje i njeni patuljci pevaju ‘guba gaba "
                "guba gaba’? ahhahah app navigation"
            ),
            (
                "video noise kako se zove 7 epizoda kada je pag boSkić nešto "
                "slavio, ovi mu uništili sve i Čvarkov imitira Šilovićku kako "
                "diriguje i njeni patuljci pevaju ‘guba gaba guba gaba’? footer"
            ),
            (
                "icons kako se zove ( epizoda kada je / boSkić nešto slavio, d "
                "ovi mu uništili sve | i Čvarkov imitira Šilovićku kako diriguje "
                "i njeni patuljci pevaju ‘guba ~ 4 gaba guba gaba’? controls"
            ),
            (
                "subtitle kako se zove fi _f epizoda kada je boSkić nešto slavio, "
                "ovi mu uništili sve i Čvarkov imitira Šilovićku kako diriguje i "
                "njeni patuljci pevaju ‘guba gaba guba gaba’? share follow"
            ),
            "unusable OCR without the stable sticker",
        ]
    )

    assert result.ok
    assert result.method == "query_cue_consensus"
    assert result.text is not None
    assert result.text.startswith("kako se zove")
    assert result.text.endswith("?")
    assert "kić nešto slavio" in result.text.casefold()
    assert "Šilovićku" in result.text
    assert "random map labels" not in result.text
    assert "share follow" not in result.text


def test_one_dropped_word_does_not_truncate_the_query_to_its_longer_half() -> None:
    """A word missing from a single frame must not cost the rest of the query.

    Word-confidence filtering removes frame noise but occasionally drops a real
    word from one frame. The stable span is contiguous, so that split would
    otherwise discard everything before the gap.
    """
    query = "Najjača je epizoda Diorama vrištao sam 20minuta od smeha umalo crkao nisam"
    result = extract_query(
        [
            query,
            query,
            # "Diorama" dips below the OCR word-confidence threshold here only.
            "Najjača je epizoda vrištao sam 20minuta od smeha umalo crkao nisam",
            query,
            query,
        ]
    )

    assert result.ok
    assert "epizoda" in result.text
    assert "Diorama" in result.text
    assert result.text.startswith("Najjača")


def test_a_single_distinctive_word_is_a_valid_query() -> None:
    """One-word comments are real queries: an episode name, or a coined word.

    "Halkidiki" names an episode; "Izvanzemljaci" is a play on "vanzemaljci".
    Both were discarded while the extractor demanded four words.
    """
    result = extract_query(
        [
            "Reply to @viewer's comment Halkidiki",
            "noise Reply to @viewer's comment Halkidiki",
            "Reply to @viewer's comment Halkidiki 7",
            "d Reply to @viewer's comment Halkidiki",
            "Reply to @viewer's comment Halkidiki i",
        ]
    )

    assert result.ok
    assert result.text == "Halkidiki"

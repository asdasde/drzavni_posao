# Organic TikTok evaluation dataset

This pipeline builds a deliberately high-precision, low-recall evaluation set
from public TikTok reply videos. It keeps a row only when both of these can be
recovered confidently:

1. the viewer's comment, extracted from the reply sticker by multi-frame OCR;
2. the canonical Državni posao episode, extracted from the post caption and
   validated against `data/youtube_episodes.json`.

It does not estimate scene boundaries. The exported judgment contains only the
YouTube episode ID, so evaluation must use episode-level relevance.

## Prerequisites

Install the optional Python dependencies, then verify the existing Tesseract
installation:

```bash
uv sync --extra tiktok-eval
tesseract --version
tesseract --list-langs
```

`eng` is required. `srp` (Serbian Cyrillic) and `srp_latn` (Serbian Latin)
should both be installed: comments appear in both scripts, and OCR runs one
pass per script group (`eng+srp_latn` and `srp`) and keeps the
higher-confidence result per frame. Only when `tesseract` is not already
available on macOS, install it once with
`brew install tesseract tesseract-lang`. On Linux, install equivalent Tesseract
and Serbian language packages through the system package manager. Verify that
each `.traineddata` file actually loads (`tesseract --list-langs` alone does
not check this); a truncated file makes Tesseract silently fall back to the
remaining languages in a combined pass.

The collector reads only public pages/posts. It does not pass cookies,
credentials, private API flags, or CAPTCHA-solving options. TikTok changes its
site frequently, so keep a browser-exported metadata manifest as the
reproducible fallback.

## 1. Discover and collect

Automatic public-profile discovery first tries yt-dlp's profile extractor and
then TikTok's public hydration HTML:

```bash
uv run --extra tiktok-eval dp-tiktok-eval collect \
  --profile-url https://www.tiktok.com/@papadubi991 \
  --limit 50
```

Profile enumeration is the least stable boundary. If TikTok or yt-dlp cannot
enumerate the account, open the public profile in a browser session that can
already display it. The following Chrome/Edge DevTools console snippet scrolls
the page and copies JSONL containing each canonical URL and the caption exposed
in the profile card:

```javascript
const posts = new Map();
let unchanged = 0;
let previousSize = -1;

while (unchanged < 5) {
  for (const anchor of document.querySelectorAll('a[href*="/video/"]')) {
    const url = new URL(anchor.href);
    const match = url.pathname.match(/^\/@[^/]+\/video\/(\d+)$/);
    if (match) {
      const alt = anchor.querySelector('img')?.alt || '';
      const caption = alt.replace(/\s+created by\s+.*$/is, '').trim();
      const existing = posts.get(match[1]);
      if (!existing || caption) {
        posts.set(match[1], {
          post_id: match[1],
          canonical_url: `${url.origin}${url.pathname}`,
          caption,
        });
      }
    }
  }
  unchanged = posts.size === previousSize ? unchanged + 1 : 0;
  previousSize = posts.size;
  window.scrollTo(0, document.body.scrollHeight);
  await new Promise((resolve) => setTimeout(resolve, 1500));
}

const rows = [...posts.values()].sort((a, b) =>
  a.canonical_url.localeCompare(b.canonical_url),
);
copy(rows.map((row) => JSON.stringify(row)).join('\n'));
console.log(`Copied ${rows.length} TikTok post records`);
```

Save the clipboard as, for example,
`artifacts/tiktok_eval/post_metadata.jsonl`, then run:

```bash
uv run --extra tiktok-eval dp-tiktok-eval collect \
  --posts-jsonl artifacts/tiktok_eval/post_metadata.jsonl
```

This route avoids a second metadata request for every post. If the browser DOM
does not expose captions, the same `rows` can be copied as a URL-only manifest:

```javascript
copy(rows.map((row) => row.canonical_url).join('\n'));
```

Save that output as `artifacts/tiktok_eval/post_urls.txt` and run:

```bash
uv run --extra tiktok-eval dp-tiktok-eval collect \
  --urls-file artifacts/tiktok_eval/post_urls.txt
```

The URL-only route asks yt-dlp to fetch each post's public metadata before OCR,
so it is more exposed to per-post rate limits than `--posts-jsonl`.

TikTok answers a large share of requests with a JS challenge page that yt-dlp
cannot parse ("Unable to extract universal data for rehydration"). Measured over
this profile, the failure is intermittent rather than per-post or a quota ban:
every post observed has both succeeded and failed across attempts, at roughly a
third success per try. So a single failure says nothing about whether a post is
reachable. `--download-attempts` (default 4) retries each post with exponential
backoff, which is what actually recovers these; `--download-delay` (default 6
seconds) additionally spaces posts apart. Set either to `0` to disable.

A download failure is still recorded as the post's newest state and supersedes
an earlier success, so re-running a whole profile while TikTok is uncooperative
can lower the accepted count even though nothing about extraction got worse.
Prefer `--retry-failed`, which leaves accepted posts untouched.

For each reply post the collector independently attempts both OCR and episode
matching. Any failure is appended to
`artifacts/tiktok_eval/collector.jsonl`, and processing continues with the next
post. Downloaded videos and sampled frames live only in temporary directories
and are removed after each attempt.

The state file is append-only and resumable. Re-running the same command skips
terminal post IDs. Use `--retry-failed` to append fresh attempts for previously
skipped posts; accepted posts remain terminal. When OCR or matching logic
changes, increment `--extractor-version` to append fresh attempts without
deleting audit history:

```bash
uv run --extra tiktok-eval dp-tiktok-eval collect \
  --posts-jsonl artifacts/tiktok_eval/post_metadata.jsonl \
  --extractor-version 2
```

Useful defaults are eight OCR frames from the first eight seconds, minimum query
confidence `0.85`, and minimum episode confidence `0.90`. They can be changed
with `--frame-count`, `--frame-seconds`, `--min-query-confidence`, and
`--min-episode-confidence`. The sticker does not always appear immediately, so a
window that is too short can miss it entirely or catch it in a single frame,
which is not enough for consensus.

Emoji inside a sticker break OCR badly: they add junk tokens and run adjacent
words together ("zivi jedna" read as "zivijedna", with the start of the sentence
lost entirely). Because a sticker is black text on white, the emoji are its only
saturated colour, so each post is read twice — once as sampled, and once with
pixels above `--emoji-saturation` (default `60`) whitened — and the stronger
consensus wins. Neither pass is better everywhere: masking rescues emoji-heavy
stickers, and loses on diacritics, whose thin colour fringing it also removes
(`uništili` degrades to `unistili`). The masked reading is only allowed to win
when it shares wording with the unmasked one, so a masked pass that erased the
text cannot install a confident artefact in its place. Set the value to `0` to
skip the second pass and halve OCR time.

A viewer's comment is sometimes a single distinctive word — an episode name
("Halkidiki"), or a coined one ("Izvanzemljaci", playing on *vanzemaljci*).
Those are real queries, so length rather than word count decides what counts as
one: `--min-query-chars` (default `8`) keeps them while still rejecting
interface text like `Ep. 23` or `super`. `--min-query-words` (default `1`)
raises the bar again if a profile's stickers are consistently longer.

OCR reads the whole frame, so the video playing behind the sticker also lands in
the text. Because the sticker is rendered text it scores far higher per word
than that background, and words below `--min-word-confidence` (default `70`,
Tesseract's 0-100 scale) are dropped along with punctuation-only tokens. This
matters more than it sounds: the background differs in every frame while the
sticker does not, so unfiltered noise prevents frames from agreeing and pushes
otherwise readable queries below the acceptance threshold. Lower the value if
genuine query words are being discarded.

## 2. Export accepted examples

Export accepted records and, when the conformed corpus exists, reject labels for
episodes that are not searchable:

```bash
uv run --extra tiktok-eval dp-tiktok-eval export \
  --segments data/segments.jsonl \
  --output artifacts/tiktok_eval/evaluation_queries.jsonl \
  --split test
```

Each row retains public source provenance but no timestamp label:

```json
{
  "query_id": "tt_<post_id>",
  "query": "ona epizoda kada se greju u kancelariji?",
  "language": "sr-Latn",
  "category": "organic_tiktok",
  "split": "test",
  "judgments": [{"youtube_video_id": "VWRBeoMbZhg", "grade": 2}],
  "provenance": {
    "platform": "tiktok",
    "post_id": "<post_id>",
    "url": "https://www.tiktok.com/@owner/video/<post_id>"
  }
}
```

Different organic comments that resolve to the same episode remain separate eval
queries. The exporter never fabricates or corrects the comment text.

### Queries awaiting a manual episode label

Most reply captions are punchlines rather than episode identifiers, so a
recovered query often has no automatic episode. Add `--review-output` to also
write those posts:

```bash
uv run --extra tiktok-eval dp-tiktok-eval export \
  --output artifacts/tiktok_eval/evaluation_queries.jsonl \
  --review-output artifacts/tiktok_eval/needs_episode_label.jsonl \
  --split test
```

Each review row carries the query, the caption, and the source URL, with
`"judgments": []` and a `review` block holding the skip status and query
confidence. Nothing in that file is ground truth: it exists so the episode can
be identified by hand. Fill in `judgments` and move a row into the main export
only after confirming the episode; leave the rest alone.

## 3. Evaluate at episode level

Run this dataset separately from interval-labeled scene evaluation:

```bash
uv run --extra search dp-evaluate \
  --queries artifacts/tiktok_eval/evaluation_queries.jsonl \
  --segments data/segments.jsonl \
  --bm25 artifacts/bm25_index.json \
  --vectors artifacts/vector_index \
  --methods bm25,dense,hybrid \
  --split test \
  --relevance-level episode \
  --output artifacts/tiktok_eval/evaluation_results.json
```

Episode relevance deduplicates retrieved segments by YouTube episode and matches
only the episode ID. Scene intervals remain required in the default
`--relevance-level scene` mode.

## Quality and policy notes

- Spot-check accepted OCR text and episode labels before freezing the benchmark.
- Freeze source URLs and the final JSONL before comparing systems; do not tune on
  the held-out `test` rows.
- Preserve failed/skipped state rows. They are useful for measuring collection
  yield without quietly converting uncertain labels into ground truth.
- A deleted/private/rate-limited post is a normal skip, not a reason to bypass an
  access control. Respect TikTok's terms, applicable privacy rules, and the
  creator/commenters' rights when retaining or sharing the resulting dataset.

const SCORE_FORMAT = new Intl.NumberFormat("sr-Latn", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});

const SECONDS_FORMAT = new Intl.NumberFormat("sr-Latn", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatScore(score: number): string {
  return SCORE_FORMAT.format(score);
}

export function formatElapsedSeconds(seconds: number): string {
  return `${SECONDS_FORMAT.format(seconds)} s`;
}

export function formatEditionDate(date: Date = new Date()): string {
  const formatted = new Intl.DateTimeFormat("sr-Latn-RS", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(date);
  const capitalized = `${formatted[0].toUpperCase()}${formatted.slice(1)}`;
  return capitalized.endsWith(".") ? capitalized : `${capitalized}.`;
}

export function formatSpeakers(speakers: string[]): string {
  return speakers.join(", ");
}

export interface TranscriptTurn {
  speaker: string | null;
  lines: string[];
}

const SPEAKER_BRACKET = /^\[([^\]]+)\]\s*(.*)$/;
const SPEAKER_COLON = /^([^\n:]{1,40}):\s+(\S.*)$/;
const INLINE_COLON_SPLIT =
  /(?=(?:^|\s)(?:[\p{Lu}]{2,}(?:\s+[\p{Lu}][\p{L}.]*){0,2}):\s+)/u;

function normalizeSpeaker(label: string): string {
  return label.replace(/\s+/g, " ").trim();
}

function isSpeakerLabel(label: string, knownSpeakers: string[]): boolean {
  const name = normalizeSpeaker(label);
  if (!name || name.length > 40) {
    return false;
  }
  if (/^SPEAKER[_\s-]?\d+$/i.test(name)) {
    return true;
  }
  const lower = name.toLowerCase();
  if (
    knownSpeakers.some((speaker) => {
      const known = speaker.toLowerCase();
      return known === lower || known.includes(lower) || lower.includes(known);
    })
  ) {
    return true;
  }
  if (/^[\p{Lu}]{2,}(?:\s+[\p{Lu}][\p{L}.]*){0,2}$/u.test(name)) {
    return true;
  }
  return /^[\p{Lu}][\p{L}'’.]+(?:\s+[\p{Lu}][\p{L}'’.]+){0,2}$/u.test(name);
}

function parsePiece(
  piece: string,
  knownSpeakers: string[],
): { speaker: string | null; text: string } {
  const trimmed = piece.trim();
  const bracket = trimmed.match(SPEAKER_BRACKET);
  if (bracket && isSpeakerLabel(bracket[1], knownSpeakers)) {
    return { speaker: normalizeSpeaker(bracket[1]), text: bracket[2].trim() };
  }
  const colon = trimmed.match(SPEAKER_COLON);
  if (colon && isSpeakerLabel(colon[1], knownSpeakers)) {
    return { speaker: normalizeSpeaker(colon[1]), text: colon[2].trim() };
  }
  return { speaker: null, text: trimmed };
}

function splitSnippet(snippet: string, knownSpeakers: string[]): string[] {
  const normalized = snippet.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return [];
  }
  if (normalized.includes("\n")) {
    return normalized.split("\n").map((line) => line.trim()).filter(Boolean);
  }
  const bracketParts = normalized
    .split(/(?=\[)/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (
    bracketParts.length > 1 &&
    bracketParts.every((part) => {
      const match = part.match(SPEAKER_BRACKET);
      return Boolean(match && isSpeakerLabel(match[1], knownSpeakers));
    })
  ) {
    return bracketParts;
  }
  const colonParts = normalized
    .split(INLINE_COLON_SPLIT)
    .map((part) => part.trim())
    .filter(Boolean);
  if (colonParts.length > 1) {
    return colonParts;
  }
  return [normalized];
}

export function parseTranscript(
  snippet: string,
  knownSpeakers: string[] = [],
): TranscriptTurn[] {
  const turns: TranscriptTurn[] = [];
  for (const piece of splitSnippet(snippet, knownSpeakers)) {
    const { speaker, text } = parsePiece(piece, knownSpeakers);
    if (!text && !speaker) {
      continue;
    }
    const last = turns.at(-1);
    if (last && last.speaker === speaker) {
      if (text) {
        last.lines.push(text);
      }
      continue;
    }
    turns.push({ speaker, lines: text ? [text] : [] });
  }
  return turns;
}

export function youtubeStartSeconds(
  url: string | null,
  clock: string,
): number {
  if (url) {
    try {
      const parsed = new URL(url);
      const raw = parsed.searchParams.get("t") ?? parsed.searchParams.get("start");
      if (raw) {
        const match = raw.match(/^(\d+)/);
        if (match) {
          return Number(match[1]);
        }
      }
    } catch {
      // Fall through to the formatted clock.
    }
  }
  const parts = clock.split(":").map(Number);
  if (parts.length === 3 && parts.every(Number.isFinite)) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  if (parts.length === 2 && parts.every(Number.isFinite)) {
    return parts[0] * 60 + parts[1];
  }
  return 0;
}

export function youtubeThumbnailSrc(videoId: string): string {
  return `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/hqdefault.jpg`;
}

export function youtubeEmbedSrc(videoId: string, startSeconds: number): string {
  const params = new URLSearchParams({
    start: String(Math.max(0, Math.round(startSeconds))),
    autoplay: "1",
    rel: "0",
  });
  return `https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}?${params}`;
}

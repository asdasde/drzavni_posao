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

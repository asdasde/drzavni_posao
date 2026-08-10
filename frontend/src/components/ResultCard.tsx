import { useState } from "react";
import type { SearchResultItem } from "../api/types";
import { formatScore, formatSpeakers } from "../lib/format";

const YOUTUBE_ICON = (
  <svg
    className="size-4 shrink-0 text-accent"
    stroke="currentColor"
    fill="none"
    strokeWidth="2"
    viewBox="0 0 24 24"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z" />
    <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02" />
  </svg>
);

interface ResultCardProps {
  result: SearchResultItem;
  isLast: boolean;
}

export default function ResultCard({ result, isLast }: ResultCardProps) {
  const top = result.rank === 1;
  const [showTranscript, setShowTranscript] = useState(result.rank <= 2);
  const [showComponents, setShowComponents] = useState(false);

  return (
    <article
      className={`flex w-full ${
        top
          ? "bg-[#f4f2ec] border-t-2 border-t-ink"
          : "border-t border-t-ink"
      } ${isLast ? "border-b-4 border-b-ink" : ""}`}
    >
      <div
        className={`flex w-[150px] flex-col shrink-0 items-center py-6 gap-1 ${
          top ? "bg-ink" : "border-r border-r-ink"
        }`}
      >
        <span
          className={`font-mono text-xs tracking-[1.2px] uppercase ${
            top ? "text-paper" : "text-neutral-600"
          }`}
        >
          No. {result.rank}
        </span>
        <span
          className={`font-display text-5xl font-black leading-none ${
            top ? "text-paper" : "text-ink"
          }`}
        >
          {formatScore(result.score)}
        </span>
        <span
          className={`font-mono text-[10px] uppercase ${
            top ? "text-neutral-400" : "text-neutral-500"
          }`}
        >
          relativni skor
        </span>
      </div>

      <div className="flex flex-col flex-1 px-8 py-6 gap-3">
        <div className="flex items-baseline gap-4">
          <h3 className="font-display text-3xl font-bold leading-[1.2] text-ink">
            {result.episode_title}
          </h3>
          <span className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
            {result.episode_id.replace(/^ep/i, "EP ")}
          </span>
        </div>

        <div className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
          Vreme u videu: {result.start_time}–{result.end_time} · Likovi:{" "}
          {formatSpeakers(result.speakers)}
        </div>

        <p className="font-body text-sm italic leading-[1.42857] text-neutral-700">
          {result.reason}
        </p>

        {showTranscript && (
          <div className="flex w-full flex-col bg-[#faf9f6] border-l-4 border-l-accent pl-5 pr-4 py-3 gap-2">
            <span className="font-mono text-[10px] uppercase text-neutral-500">
              Transkript — dokaz
            </span>
            <p className="max-w-[672px] font-body text-base leading-relaxed text-justify text-ink">
              {result.snippet}
            </p>
          </div>
        )}

        {showComponents && (
          <div className="flex flex-wrap gap-x-6 gap-y-1 border border-ink px-4 py-3 font-mono text-xs text-ink">
            {Object.entries(result.score_components).map(([name, value]) => (
              <span key={name}>
                <span className="uppercase tracking-[1.2px] text-neutral-600">
                  {name}:
                </span>{" "}
                {formatScore(value)}
              </span>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center pt-1 gap-4">
          {result.youtube_url && (
            <a
              href={result.youtube_url}
              target="_blank"
              rel="noreferrer"
              className="flex items-center border border-ink px-4 py-2 gap-2"
            >
              {YOUTUBE_ICON}
              <span className="font-mono text-xs tracking-[1.2px] uppercase text-ink">
                Pogledaj od {result.start_time}
              </span>
            </a>
          )}
          {!showTranscript && (
            <button
              type="button"
              onClick={() => setShowTranscript(true)}
              className="flex items-center gap-2 cursor-pointer"
            >
              <svg
                className="size-4 shrink-0 text-neutral-600"
                stroke="currentColor"
                fill="none"
                strokeWidth="2"
                viewBox="0 0 24 24"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <line x1="12" y1="8" x2="12" y2="16" />
                <line x1="8" y1="12" x2="16" y2="12" />
              </svg>
              <span className="font-mono text-xs tracking-[1.2px] uppercase underline decoration-black underline-offset-4 text-neutral-600">
                Prikaži transkript
              </span>
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowComponents((current) => !current)}
            className="font-mono text-xs tracking-[1.2px] uppercase underline decoration-black underline-offset-4 text-neutral-600 cursor-pointer"
          >
            Komponente skora
          </button>
        </div>
      </div>
    </article>
  );
}

import { useState } from "react";
import type { SearchResultItem } from "../api/types";
import {
  formatScore,
  formatSpeakers,
  youtubeEmbedSrc,
  youtubeStartSeconds,
  youtubeThumbnailSrc,
} from "../lib/format";
import Transcript from "./Transcript";

const PLAY_ICON = (
  <svg
    className="size-5 shrink-0"
    stroke="currentColor"
    fill="currentColor"
    viewBox="0 0 24 24"
  >
    <polygon points="8,5 20,12 8,19" />
  </svg>
);

interface ResultCardProps {
  result: SearchResultItem;
  isLast: boolean;
}

export default function ResultCard({ result, isLast }: ResultCardProps) {
  const top = result.rank === 1;
  const [showComponents, setShowComponents] = useState(false);
  const [showPlayer, setShowPlayer] = useState(false);
  const [thumbFailed, setThumbFailed] = useState(false);
  const startSeconds = youtubeStartSeconds(result.youtube_url, result.start_time);
  const embedSrc = result.youtube_video_id
    ? youtubeEmbedSrc(result.youtube_video_id, startSeconds)
    : null;
  const thumbSrc = result.youtube_video_id
    ? youtubeThumbnailSrc(result.youtube_video_id)
    : null;

  return (
    <article
      className={`flex w-full ${
        top ? "bg-[#f4f2ec] border-t-2 border-t-ink" : "border-t border-t-ink"
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

      <div className="grid min-w-0 flex-1 grid-cols-1 gap-5 px-8 py-6 lg:grid-cols-[minmax(0,1fr)_minmax(280px,42%)] lg:gap-8">
        <div className="flex min-w-0 flex-col gap-3 lg:col-start-1 lg:row-start-1">
          <div className="flex items-baseline gap-4">
            <h3 className="font-display text-3xl font-bold leading-[1.2] text-ink">
              {result.episode_title}
            </h3>
            {result.episode_number != null && (
              <span className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
                Ep. {result.episode_number}
              </span>
            )}
          </div>
          <div className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
            Vreme u videu: {result.start_time}–{result.end_time} · Likovi:{" "}
            {formatSpeakers(result.speakers)}
          </div>
        </div>

        {embedSrc && (
          <aside className="flex w-full flex-col gap-2 lg:col-start-2 lg:row-span-3 lg:row-start-1 lg:self-center">
            <div className="aspect-video w-full border-2 border-ink bg-ink">
              {showPlayer ? (
                <iframe
                  title={`${result.episode_title} od ${result.start_time}`}
                  src={embedSrc}
                  className="size-full"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setShowPlayer(true)}
                  className="relative flex size-full cursor-pointer items-center justify-center overflow-hidden"
                >
                  {thumbSrc && !thumbFailed ? (
                    <img
                      src={thumbSrc}
                      alt=""
                      onError={() => setThumbFailed(true)}
                      className="absolute inset-0 size-full object-cover opacity-80"
                    />
                  ) : null}
                  <span className="relative flex items-center gap-2 bg-paper px-4 py-2 font-mono text-xs tracking-[1.2px] uppercase text-ink">
                    {PLAY_ICON}
                    Pogledaj od {result.start_time}
                  </span>
                </button>
              )}
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              {result.youtube_url && (
                <a
                  href={result.youtube_url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-mono text-xs tracking-[1.2px] uppercase underline decoration-black underline-offset-4 text-neutral-600"
                >
                  Otvori na YouTube-u
                </a>
              )}
              {showPlayer && (
                <button
                  type="button"
                  onClick={() => setShowPlayer(false)}
                  className="font-mono text-xs tracking-[1.2px] uppercase underline decoration-black underline-offset-4 text-neutral-600 cursor-pointer"
                >
                  Sakrij video
                </button>
              )}
            </div>
          </aside>
        )}

        <div className="flex min-w-0 flex-col gap-3 lg:col-start-1 lg:row-start-2">
          <Transcript snippet={result.snippet} speakers={result.speakers} />

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
            <button
              type="button"
              onClick={() => setShowComponents((current) => !current)}
              className="font-mono text-xs tracking-[1.2px] uppercase underline decoration-black underline-offset-4 text-neutral-600 cursor-pointer"
            >
              Komponente skora
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

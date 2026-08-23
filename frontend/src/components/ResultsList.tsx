import type { SearchResponse } from "../api/types";
import { METHOD_LABELS } from "../api/types";
import { formatElapsedSeconds } from "../lib/format";
import ResultCard from "./ResultCard";

interface ResultsListProps {
  response: SearchResponse;
}

export default function ResultsList({ response }: ResultsListProps) {
  return (
    <section className="flex w-full flex-col">
      <div className="flex w-full justify-between items-end pt-10 pb-4 px-10">
        <div className="flex items-baseline gap-3">
          <h2 className="font-display text-4xl font-black leading-[1.11111] tracking-[-0.9px] text-ink">
            Najverovatniji rezultati
          </h2>
          <span className="font-mono text-sm text-accent">
            ({response.results.length})
          </span>
        </div>
        <span className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
          Upit: „{response.query}“ · {METHOD_LABELS[response.method]} ·{" "}
          {formatElapsedSeconds(response.took_seconds)}
        </span>
      </div>
      {response.results.length === 0 ? (
        <div className="border-t-2 border-t-ink px-10 py-10 font-body text-lg italic text-neutral-600">
          Nema rezultata — pokušajte da opišete scenu drugim rečima.
        </div>
      ) : (
        response.results.map((result, index) => (
          <ResultCard
            key={`${response.method}:${result.segment_id}`}
            result={result}
            isLast={index === response.results.length - 1}
          />
        ))
      )}
    </section>
  );
}

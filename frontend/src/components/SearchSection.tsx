import { useState, type FormEvent } from "react";
import {
  METHOD_HINTS,
  METHOD_LABELS,
  METHOD_TECHNICAL_LABELS,
  type SearchMethod,
  type SearchParams,
} from "../api/types";

const EXAMPLE_QUERIES = [
  "kad Boškić nema keš za kafu jer ne primaju kartice",
  "kad Torbica hoće da se prijavi za Prvi glas Srbije",
  "Čvarkov sakrio ušteđevinu u stare registratore",
];

const SIMPLE_METHODS: SearchMethod[] = ["bm25", "dense"];
const TECHNICAL_METHODS: SearchMethod[] = ["bm25", "dense", "hybrid", "hybrid_rerank"];

function methodButtonClass(selected: boolean) {
  return `border-b border-r border-ink px-5 py-2 uppercase tracking-[1.2px] cursor-pointer ${
    selected ? "bg-ink text-paper" : "bg-transparent text-ink"
  }`;
}

interface SearchSectionProps {
  query: string;
  onQueryChange: (query: string) => void;
  method: SearchMethod;
  onMethodChange: (method: SearchMethod) => void;
  params: Pick<SearchParams, "top_k" | "candidate_k" | "group_by_episode">;
  loading: boolean;
  onSearch: (query: string) => void;
}

export default function SearchSection({
  query,
  onQueryChange,
  method,
  onMethodChange,
  params,
  loading,
  onSearch,
}: SearchSectionProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSearch(query);
  };

  return (
    <section className="flex w-full flex-col items-center bg-section border-b border-b-ink px-10 py-14 gap-6">
      <div className="flex flex-col items-center gap-3">
        <span className="bg-accent px-3 py-1 font-mono text-xs tracking-[1.2px] uppercase text-paper">
          Potraga u toku
        </span>
        <h2 className="font-display text-5xl font-black leading-none tracking-[-1.2px] text-center text-ink">
          Koja je ono epizoda bila…?
        </h2>
        <p className="max-w-[576px] font-body text-lg leading-[1.55556] text-center text-neutral-600">
          Opišite scenu svojim rečima — dobićete epizodu, tačan minutaž i
          transkript kao dokaz.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="flex w-full max-w-[820px] bg-paper border-2 border-ink shadow-[6px_6px_0px_rgb(17,17,17)]"
      >
        <div className="flex items-center flex-1 px-5 py-4 gap-3">
          <svg
            className="size-5 shrink-0"
            stroke="currentColor"
            fill="none"
            strokeWidth="2"
            viewBox="0 0 24 24"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={EXAMPLE_QUERIES[0]}
            className="flex-1 bg-transparent font-mono text-sm text-ink placeholder:text-neutral-500 focus:outline-none"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="shrink-0 bg-ink px-8 font-mono text-sm tracking-[1.4px] uppercase text-paper disabled:opacity-70 cursor-pointer"
        >
          {loading ? "Tražim…" : "Pretraži"}
        </button>
      </form>

      <div className="flex w-full max-w-[820px] flex-col items-center gap-3">
        <div className="flex flex-wrap items-center justify-center gap-4 font-mono text-xs tracking-[1.2px] uppercase">
          <span className="text-neutral-600">Način:</span>
          <div className="flex border-t border-l border-ink">
            {SIMPLE_METHODS.map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => onMethodChange(value)}
                className={methodButtonClass(method === value)}
              >
                {METHOD_LABELS[value]}
              </button>
            ))}
            <div
              className={`flex ${
                method === "hybrid" || method === "hybrid_rerank" ? "bg-neutral-200" : ""
              }`}
            >
              <button
                type="button"
                onClick={() => onMethodChange("hybrid")}
                className={methodButtonClass(method === "hybrid")}
              >
                Kombinovano
              </button>
              <button
                type="button"
                title="Preciznija, ali sporija varijanta — za maksimalno razumevanje"
                onClick={() => onMethodChange("hybrid_rerank")}
                className={methodButtonClass(method === "hybrid_rerank")}
              >
                Preciznije
              </button>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setAdvancedOpen((open) => !open)}
            aria-expanded={advancedOpen}
            className={`border border-ink px-3 py-2 uppercase tracking-[1.2px] cursor-pointer ${
              advancedOpen ? "bg-ink text-paper" : "bg-transparent text-ink"
            }`}
          >
            Napredno {advancedOpen ? "▴" : "▾"}
          </button>
        </div>

        <p className="font-body text-sm italic text-neutral-600 text-center normal-case tracking-normal">
          {METHOD_HINTS[method]}
        </p>

        {advancedOpen && (
          <div className="flex w-full flex-col gap-3 border border-ink bg-paper px-5 py-4">
            <div className="flex flex-wrap items-center gap-3 font-mono text-xs tracking-[1.2px] uppercase">
              <span className="text-neutral-600">Algoritam:</span>
              <div className="flex border-t border-l border-ink">
                {TECHNICAL_METHODS.map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => onMethodChange(value)}
                    className={`border-b border-r border-ink px-4 py-2 uppercase tracking-[1.2px] cursor-pointer ${
                      method === value ? "bg-ink text-paper" : "bg-transparent text-ink"
                    }`}
                  >
                    {METHOD_TECHNICAL_LABELS[value]}
                  </button>
                ))}
              </div>
            </div>
            <p className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
              {METHOD_LABELS[method]} = {METHOD_TECHNICAL_LABELS[method]} ·
              Rezultata: {params.top_k} · Kandidata: {params.candidate_k} ·{" "}
              {params.group_by_episode ? "1 scena / epizodi" : "sve scene"}
            </p>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <span className="font-mono text-xs tracking-[1.2px] uppercase text-neutral-600">
          Primeri:
        </span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              onQueryChange(example);
              onSearch(example);
            }}
            className="font-body text-sm text-ink underline decoration-accent decoration-2 underline-offset-4 cursor-pointer"
          >
            {example}
          </button>
        ))}
      </div>
    </section>
  );
}

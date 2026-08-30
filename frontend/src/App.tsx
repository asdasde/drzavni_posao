import { useRef, useState } from "react";
import { search } from "./api/client";
import type { SearchMethod, SearchResponse } from "./api/types";
import FooterTicker from "./components/FooterTicker";
import Masthead from "./components/Masthead";
import ResultsList from "./components/ResultsList";
import SearchSection from "./components/SearchSection";

const SEARCH_PARAMS = {
  top_k: 5,
  candidate_k: 50,
  group_by_episode: true,
};

export default function App() {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState<SearchMethod>("hybrid");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  const runSearch = async (searchQuery: string, searchMethod: SearchMethod = method) => {
    const trimmed = searchQuery.trim();
    if (!trimmed || loading) {
      return;
    }
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const result = await search({
        query: trimmed,
        method: searchMethod,
        ...SEARCH_PARAMS,
      });
      if (id === requestId.current) {
        setResponse(result);
      }
    } catch (cause) {
      if (id === requestId.current) {
        setError(cause instanceof Error ? cause.message : "Pretraga nije uspela.");
        setResponse(null);
      }
    } finally {
      if (id === requestId.current) {
        setLoading(false);
      }
    }
  };

  const handleMethodChange = (nextMethod: SearchMethod) => {
    setMethod(nextMethod);
    if (response) {
      void runSearch(response.query, nextMethod);
    }
  };

  return (
    <div className="flex min-h-screen w-full flex-col bg-paper text-ink">
      <Masthead />
      <SearchSection
        query={query}
        onQueryChange={setQuery}
        method={method}
        onMethodChange={handleMethodChange}
        params={SEARCH_PARAMS}
        loading={loading}
        onSearch={(value) => void runSearch(value)}
      />
      {error && (
        <div className="mx-10 mt-10 border-l-4 border-l-accent bg-[#faf9f6] px-5 py-3">
          <span className="font-mono text-[10px] uppercase text-neutral-500 block">
            Greška
          </span>
          <p className="font-body text-base italic text-ink">{error}</p>
        </div>
      )}
      {response && !error && <ResultsList response={response} />}
      <FooterTicker />
    </div>
  );
}

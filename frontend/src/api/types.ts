// Mirrors SearchResult.to_dict() in drzavni_posao/search/retrieval/models.py
// and the parameters of SearchEngine.search().

export type SearchMethod = "bm25" | "dense" | "hybrid" | "hybrid_rerank";

export const METHOD_LABELS: Record<SearchMethod, string> = {
  bm25: "BM25",
  dense: "Dense",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + reranker",
};

export interface SearchParams {
  query: string;
  method: SearchMethod;
  top_k: number;
  candidate_k: number;
  group_by_episode: boolean;
}

export interface SearchResultItem {
  rank: number;
  youtube_video_id: string;
  episode_title: string;
  segment_id: string;
  start_seconds: number;
  end_seconds: number;
  start_time: string;
  end_time: string;
  transcript_start_time: string;
  transcript_end_time: string;
  youtube_url: string | null;
  speakers: string[];
  snippet: string;
  score: number;
  score_components: Record<string, number>;
  matched_characters: string[];
  matched_terms: string[];
  reason: string;
}

export interface SearchResponse {
  query: string;
  method: SearchMethod;
  took_seconds: number;
  results: SearchResultItem[];
}

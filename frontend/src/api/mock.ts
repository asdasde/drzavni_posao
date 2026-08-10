import type { SearchParams, SearchResponse, SearchResultItem } from "./types";

const MOCK_LATENCY_MS = 420;

const MOCK_RESULTS: SearchResultItem[] = [
  {
    rank: 1,
    episode_id: "ep143",
    episode_title: "Odlaganje izbora",
    segment_id: "ep143_0007",
    start_seconds: 436,
    end_seconds: 527,
    start_time: "07:41",
    end_time: "09:12",
    transcript_start_time: "07:16",
    transcript_end_time: "08:47",
    youtube_url: "https://www.youtube.com/watch?v=dp-ep143&t=461s",
    speakers: ["Boškić", "Čvarkov", "Torbica"],
    snippet:
      "ČVARKOV: Boškiću, izbori se ne odlažu, izbori se organizovano ne održe! BOŠKIĆ: A ko će to da potpiše? ČVARKOV: Onaj ko je potpisao i prošli put — niko.",
    score: 0.912,
    score_components: {
      bm25: 0.94,
      dense: 0.9,
      metadata: 1.0,
      bm25_raw: 14.21,
      dense_raw: 0.84,
    },
    matched_characters: ["boskic", "cvarkov"],
    matched_terms: ["izbori", "svađa"],
    reason:
      "Zašto: poklapanje ključnih reči „izbori“ i imena likova; visok semantički skor scene.",
  },
  {
    rank: 2,
    episode_id: "ep087",
    episode_title: "Glasačka kutija",
    segment_id: "ep087_0003",
    start_seconds: 160,
    end_seconds: 241,
    start_time: "03:05",
    end_time: "04:26",
    transcript_start_time: "02:40",
    transcript_end_time: "04:01",
    youtube_url: "https://www.youtube.com/watch?v=dp-ep087&t=185s",
    speakers: ["Čvarkov", "Torbica"],
    snippet:
      "TORBICA: [Преко воље] Čvarkove, gde je kutija? ČVARKOV: Koja kutija? TORBICA: Glasačka. ČVARKOV: Služi kao saksija, od marta.",
    score: 0.847,
    score_components: {
      bm25: 0.91,
      dense: 0.78,
      metadata: 0.5,
      bm25_raw: 12.03,
      dense_raw: 0.79,
    },
    matched_characters: ["cvarkov"],
    matched_terms: ["glasanje", "komisija"],
    reason:
      "Zašto: BM25 pogodak na „izborna komisija“; scena pominje glasanje u kancelariji.",
  },
  {
    rank: 3,
    episode_id: "ep211",
    episode_title: "Referendum u kancelariji",
    segment_id: "ep211_0011",
    start_seconds: 683,
    end_seconds: 757,
    start_time: "11:48",
    end_time: "13:02",
    transcript_start_time: "11:23",
    transcript_end_time: "12:37",
    youtube_url: "https://www.youtube.com/watch?v=dp-ep211&t=708s",
    speakers: ["Boškić", "Torbica"],
    snippet:
      "BOŠKIĆ: Predlažem referendum: ostajemo li danas prekovremeno? TORBICA: Ja glasam nevažeće. BOŠKIĆ: Upisano — jednoglasno usvojeno.",
    score: 0.791,
    score_components: {
      bm25: 0.62,
      dense: 0.88,
      metadata: 0.5,
      bm25_raw: 7.44,
      dense_raw: 0.76,
    },
    matched_characters: ["boskic"],
    matched_terms: ["glasanje"],
    reason: "Zašto: semantička sličnost sa „svađa oko glasanja“; niži leksički skor.",
  },
];

export async function mockSearch(params: SearchParams): Promise<SearchResponse> {
  await new Promise((resolve) => setTimeout(resolve, MOCK_LATENCY_MS));
  return {
    query: params.query,
    method: params.method,
    took_seconds: MOCK_LATENCY_MS / 1000,
    results: MOCK_RESULTS.slice(0, params.top_k),
  };
}

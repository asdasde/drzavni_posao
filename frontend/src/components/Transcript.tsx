import { parseTranscript } from "../lib/format";

interface TranscriptProps {
  snippet: string;
  speakers: string[];
}

function DialogueLine({ text }: { text: string }) {
  const staged = text.match(/^\[([^\]]+)\]\s*(.*)$/);
  if (staged) {
    return (
      <p className="font-body text-base leading-relaxed text-ink">
        <em className="text-neutral-500">[{staged[1]}]</em>
        {staged[2] ? ` ${staged[2]}` : ""}
      </p>
    );
  }
  return <p className="font-body text-base leading-relaxed text-ink">{text}</p>;
}

export default function Transcript({ snippet, speakers }: TranscriptProps) {
  const turns = parseTranscript(snippet, speakers);
  if (turns.length === 0) {
    return null;
  }

  return (
    <div className="flex w-full flex-col bg-[#faf9f6] border-l-4 border-l-accent px-5 py-4 gap-4">
      <span className="font-mono text-[10px] uppercase tracking-[1.2px] text-neutral-500">
        Transkript scene
      </span>
      <div className="flex flex-col gap-4">
        {turns.map((turn, index) => (
          <div
            key={`${turn.speaker ?? "scena"}-${index}`}
            className="grid grid-cols-1 gap-1 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-x-5"
          >
            <span
              className={`font-mono text-xs tracking-[1.2px] uppercase sm:pt-0.5 ${
                turn.speaker ? "text-accent" : "text-neutral-400"
              }`}
            >
              {turn.speaker ?? "Scena"}
            </span>
            <div className="flex flex-col gap-2">
              {turn.lines.map((line, lineIndex) => (
                <DialogueLine key={lineIndex} text={line} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

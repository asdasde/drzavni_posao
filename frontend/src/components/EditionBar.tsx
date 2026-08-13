import { formatEditionDate } from "../lib/format";

interface EditionBarProps {
  episodeCount?: number | null;
}

export default function EditionBar({ episodeCount }: EditionBarProps) {
  return (
    <div className="flex w-full justify-between items-center border-b border-b-ink px-10 py-2 font-mono text-xs tracking-[1.2px] uppercase">
      <div className="flex-1 text-left text-ink">{formatEditionDate()}</div>
      <div className="flex-1 text-center text-neutral-600">
        Službeni pretraživač scena — Novi Sad
      </div>
      <div className="flex-1 text-right text-ink">
        {episodeCount ? `${episodeCount} epizoda` : ""}
      </div>
    </div>
  );
}

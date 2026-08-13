interface FooterTickerProps {
  episodeCount?: number | null;
}

export default function FooterTicker({ episodeCount }: FooterTickerProps) {
  return (
    <footer className="mt-auto flex w-full bg-ink">
      <div className="flex w-full justify-between items-center px-10 py-3 font-mono text-xs tracking-[1.2px] uppercase">
        <div className="flex items-center gap-6">
          <span className="text-paper">
            {episodeCount ? `${episodeCount} epizoda` : "Arhiva scena"}
          </span>
          <span className="text-neutral-400 tracking-normal">✧</span>
          <span className="text-neutral-400">
            Pretraga po smislu — pronađi scenu i kad ne znaš tačne reči
          </span>
        </div>
        <span className="text-neutral-400">Štampano u Novom Sadu</span>
      </div>
    </footer>
  );
}

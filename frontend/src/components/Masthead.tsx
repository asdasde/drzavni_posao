import EditionBar from "./EditionBar";

export default function Masthead() {
  return (
    <header className="flex w-full flex-col border-b-4 border-b-ink">
      <EditionBar />
      <div className="flex w-full flex-col items-center py-8 gap-2">
        <h1 className="font-display text-7xl font-black leading-[0.9] tracking-[-3.6px] text-ink">
          DRŽAVNI POSAO
        </h1>
        <p className="font-body text-lg italic text-neutral-600">
          Arhiva scena · pretraga po opisu, na srpskom ili engleskom
        </p>
      </div>
    </header>
  );
}

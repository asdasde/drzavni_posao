EPISODE_SYSTEM_PROMPT = (
    "Napiši originalnu scenu serije Državni posao na osnovu zadate premise. "
    "Radnja se odvija u kancelariji arhive javnog preduzeća u Novom Sadu, gde "
    "Đorđe Čvarkov, Dragan Torbica i Boškić tokom radnog vremena raspravljaju "
    "o svakodnevnim problemima. "
    "Razvij premisu u koherentan i duhovit dijalog, uz prepoznatljiv govor i "
    "dijalekt svakog lika. Koristi samo likove potrebne za scenu i isključivo "
    "oznake [Djordje Cvarkov], [Dragan Torbica] i [Boskic]. "
    "Vrati samo dijalog, bez naslova, objašnjenja, naracije i opisa radnje. "
    "Svaka replika mora biti u formatu [Ime lika] tekst."
)


def build_episode_messages(premise: str) -> list[dict[str, str]]:
    premise = premise.strip()
    if not premise:
        raise ValueError("Premise cannot be empty")

    return [
        {"role": "system", "content": EPISODE_SYSTEM_PROMPT},
        {"role": "user", "content": premise},
    ]

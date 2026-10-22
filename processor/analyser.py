def analyze_text(text: str) -> dict:
    statistics = {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
    }

    return statistics

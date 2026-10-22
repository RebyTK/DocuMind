from processor.analyser import analyze_text


def test_analyze_text_returns_correct_statistics():
    text = "Hi there\nAI rocks"

    result = analyze_text(text)

    assert result == {
        "characters": 17,
        "words": 4,
        "lines": 2,
    }

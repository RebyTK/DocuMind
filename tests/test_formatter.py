from processor.formatter import format_elements_as_markdown


def test_format_elements_as_markdown():
    elements = [
        {
            "type": "heading",
            "text": "Quarterly Revenue",
            "style": "Heading 1",
        },
        {
            "type": "list_item",
            "text": "Revenue increased by 12%",
        },
        {
            "type": "table",
            "rows": [
                ["Quarter", "Revenue"],
                ["Q1", "$100,000"],
            ],
        },
    ]

    result = format_elements_as_markdown(elements)

    assert "# Quarterly Revenue" in result
    assert "- Revenue increased by 12%" in result
    assert "| Quarter | Revenue |" in result
    assert "| --- | --- |" in result
    assert "| Q1 | $100,000 |" in result
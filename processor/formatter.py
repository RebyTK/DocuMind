def format_elements_as_markdown(elements: list[dict]) -> str:
    """Convert structured document elements into Markdown."""
    formatted_parts = []

    for element in elements:
        element_type = element["type"]

        if element_type == "heading":
            text = element["text"]
            style = element.get("style", "Heading 1")

            try:
                level = int(style.split()[-1])
            except (ValueError, IndexError):
                level = 1

            level = max(1, min(level, 6))
            formatted_parts.append(f'{"#" * level} {text}')

        elif element_type == "paragraph":
            formatted_parts.append(element["text"])

        elif element_type == "list_item":
            formatted_parts.append(f'- {element["text"]}')

        elif element_type == "table":
            rows = element["rows"]

            if not rows:
                continue

            column_count = max(len(row) for row in rows)
            normalized_rows = [
                row + [""] * (column_count - len(row))
                for row in rows
            ]

            header = normalized_rows[0]
            separator = ["---"] * column_count

            markdown_rows = [
                "| " + " | ".join(row) + " |"
                for row in [header, separator, *normalized_rows[1:]]
            ]

            formatted_parts.append("\n".join(markdown_rows))

    return "\n\n".join(formatted_parts)
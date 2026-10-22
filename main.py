import logging
import argparse
import json
from pathlib import Path

from processor.cleaner import clean_text
from processor.analyser import analyze_text
from processor.pdf_reader import extract_text_from_pdf
from processor.docx_reader import extract_text_from_docx
from processor.formatter import format_elements_as_markdown
from processor.pdf_table_reader import extract_tables_from_pdf


def main():
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
            "output/documind.log",
            encoding="utf-8",
                ),
            ],
        )
    logging.info("DocuMind Started!!")

    parser = argparse.ArgumentParser(
        description="DocuMind - AI Document Processor"
    )
    parser.add_argument(
        "file",
        help="Path to the document to process",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.is_file():
        logging.error("File not found: %s", file_path)
        return
    
    supported_extensions = {".txt", ".pdf",".docx"}
    if file_path.suffix.lower() not in supported_extensions:
        logging.error(
            "Unsupported file type: %s",
            file_path.suffix or "(no extension)",
        )
        return

    try:
        structured_elements = None
        pdf_tables = []

        if file_path.suffix.lower() == ".pdf":
            content = extract_text_from_pdf(file_path)
            pdf_tables = extract_tables_from_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            docx_data = extract_text_from_docx(file_path)
            structured_elements = docx_data["elements"]
            content = format_elements_as_markdown(structured_elements)
        else:
            content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        logging.error(
            "Unable to decode %s as UTF-8: %s",
            file_path,
            error,
        )
        return
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return
    except OSError as error:
        logging.error(
            "Unable to read file %s: %s",
            file_path,
            error,
        )
        return
    except Exception:
        logging.exception("Unexpected error while reading %s", file_path)
        return

    
    if pdf_tables:
        table_sections = []

        for table_data in pdf_tables:
            rows = table_data["rows"]

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

            table_markdown = "\n".join(markdown_rows)

            table_sections.append(
                f"## Table {table_data['table_number']} "
                f"(Page {table_data['page']})\n\n{table_markdown}"
            )

        if table_sections:
            content = content + "\n\n" + "\n\n".join(table_sections)


    if file_path.suffix.lower() == ".docx" or pdf_tables:
        cleaned_content = content
    else:
        cleaned_content = clean_text(content)

    stats = analyze_text(cleaned_content)
    print("================================")
    print("AI DOCUMENT PROCESSOR")
    print("================================")
    print("File:", file_path.name)
    print("Type:", file_path.suffix)
    print("\nDocument Statistics")

    print("Characters:", stats["characters"])
    print("Words:", stats["words"])
    print("Lines:", stats["lines"])
    logging.info(
        "Document analyzed: %s (%d words)",
        file_path.name,
        stats["words"],
    )

    
    output_extension = (
       ".md" if file_path.suffix.lower() == ".docx" else ".txt"
    )

    output_path = output_dir / (
        file_path.stem + "_cleaned" + output_extension
    )

    try:
        output_path.write_text(
        cleaned_content,
        encoding="utf-8",
        )
    except OSError as error:
        logging.error(
            "Unable to write metadata to %s: %s",
            output_path,
            error,
        )      
        return

    print("Cleaned content written to:", output_path)

    metadata = {
        "filename" : file_path.name,
        "file_type": file_path.suffix,
        "output_file": output_path.name,
        "characters": stats["characters"],
        "words": stats["words"],
        "lines": stats["lines"]
    }

    if structured_elements is not None:
        metadata["elements"] = structured_elements
    if pdf_tables:
        metadata["tables"] = pdf_tables

    output_file = output_dir / (
        file_path.stem + "_metadata.json"
        )
    try:
        with output_file.open("w",encoding="utf-8") as file:
            json.dump(
                metadata, 
                file, 
                indent=4, 
                ensure_ascii=False
            )
    except OSError as error:
        logging.error(
        "Unable to write metadata to %s: %s",
        output_file,
        error,
        )
        return
    print("Metadata written to: ", output_file)

if __name__ == "__main__":
    main()








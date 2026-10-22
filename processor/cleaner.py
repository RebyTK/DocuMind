def clean_text(text:str)-> str:
    #Remove whitespaces at beginning/end and normalize multiple spaces to a single space
    if not isinstance(text,str):
        raise TypeError("Text must be a String")
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            cleaned_lines.append(' '.join(line.split()))
    return '\n'.join(cleaned_lines)
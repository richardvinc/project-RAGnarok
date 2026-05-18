import ijson
import re
import unicodedata
from pathlib import Path
import html

INPUT_FILE = "0b64cbd8-7497-4756-b86f-8924332da60b.json"
OUTPUT_DIR = Path("documents")


def clean_wikipedia_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)

    # Remove common Wikipedia error/artifact messages
    text = re.sub(
        r"Citation error\. See inline comment how to fix\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove numeric citations like [1], [14], [23]
    text = re.sub(r"\[\d+\]", "", text)

    # Remove citation-needed markers
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    # Convert Wikipedia headings to Markdown headings
    def heading_to_md(match):
        equals = match.group(1)
        heading = match.group(2).strip()
        level = min(len(equals), 6)
        return "\n" + ("#" * level) + f" {heading}\n"

    text = re.sub(r"(={2,6})([^=]+)\1", heading_to_md, text)

    # Remove low-value terminal sections for RAG
    text = re.sub(
        r"\n## (Notes|References|External links and further reading)\b.*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Clean empty bullets
    text = re.sub(r"(?m)^\s*\*\s*\*\s*$", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")
    title = title.lower()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"[\s_-]+", "-", title)
    title = title.strip("-")
    return title or "untitled"


def safe_filename(item_id: str, title: str) -> str:
    normalized_title = normalize_title(title)
    normalized_id = re.sub(r"[^\w.-]", "_", str(item_id).strip())
    return f"{normalized_id}_{normalized_title}.md"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(INPUT_FILE, "rb") as f:
        items = ijson.items(f, "item")

        for item in items:
            item_id = item.get("id", "")
            title = item.get("title", "")
            text = clean_wikipedia_text(item.get("text", ""))

            filename = safe_filename(item_id, title)
            path = OUTPUT_DIR / filename

            markdown = f"# {title}\n\n{text}\n"

            with open(path, "w", encoding="utf-8") as out:
                out.write(markdown)

            count += 1

    print(f"Created {count} markdown files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
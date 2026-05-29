from __future__ import annotations

from apps.api.config import settings
from apps.api.services.retrieval_service import ingest_directory


def main() -> None:
    results = ingest_directory(settings.documents_dir)
    for result in results:
        print(f"Ingested {result.file_name}: {result.chunks_created} chunks")


if __name__ == "__main__":
    main()

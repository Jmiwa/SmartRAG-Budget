"""
Load production JSON files from the `dbjson` folder and upsert them as vector data
into the Pinecone index.

Main steps:
1. Extract "consultation_text" from each JSON and generate embeddings with OpenAI.
2. Store the remaining fields (e.g., "financial_data", "expert_reduction") as metadata.
3. To avoid OpenAI API rate limits (TPM), split processing into batches of BATCH_SIZE
   and wait SLEEP_TIME between batches.
"""

import json
import os
import sys
import time  # For time.sleep()
from pathlib import Path
from typing import Iterable, List, Tuple

# Avoid clashing with the installed `pinecone` package
_CURRENT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(_CURRENT_DIR))
except ValueError:
    pass
sys.path.append(str(_CURRENT_DIR))

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None

from langchain_core.documents import Document  # type: ignore
from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_pinecone import PineconeVectorStore  # type: ignore

# Recommended for pinecone-client v3.x and later
# (For v2.x use `from pinecone import Pinecone, ServerlessSpec`)
try:
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except ImportError:
    print("Error: pinecone-client is not installed.")
    print("pip install pinecone-client")
    sys.exit(1)


# --- 1. Settings for v3 ---
INDEX_NAME = "your_index_name"  # Set your new index name
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# --- 2. Rate limit mitigation constants ---
BATCH_SIZE = 100
SLEEP_TIME = 10


def require_env(var_name: str) -> str:
    """Return the value of the requested environment variable or raise an error."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"{var_name} environment variable is not set.")
    return value


def iter_json_files(directory: Path) -> Iterable[Path]:
    """Yield JSON file paths under the specified directory in sorted order."""
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path


def load_documents(data_dir: Path) -> Tuple[List[str], List[Document]]:
    """
    Load JSON files and convert them into LangChain documents.
    - `consultation_text` is used for the vector (page_content).
    - All other top-level keys are stored in metadata.
    """
    ids: List[str] = []
    documents: List[Document] = []

    print(f"Loading documents from {data_dir}...")

    all_files = list(iter_json_files(data_dir))
    print(f"Found {len(all_files)} JSON files.")

    for json_path in all_files:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        # 1. Field to embed (page_content)
        content = payload.get("consultation_text")
        if not content:
            print(
                f"Warning: 'consultation_text' not found in {json_path.name}. Skipping."
            )
            continue

        # 2. Pinecone vector ID
        vector_id = str(payload.get("id"))
        if not vector_id:
            print(f"Warning: 'id' not found in {json_path.name}. Skipping.")
            continue

        # 3. Metadata
        metadata_payload = payload.copy()
        metadata_payload.pop("consultation_text", None)
        metadata_payload.pop("id", None)

        stringified_metadata = {"source": json_path.name, "id": vector_id}
        for key, value in metadata_payload.items():
            if isinstance(value, (dict, list)):
                stringified_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool, type(None))):
                stringified_metadata[key] = value
            else:
                stringified_metadata[key] = str(value)

        ids.append(vector_id)
        documents.append(Document(page_content=content, metadata=stringified_metadata))

    return ids, documents


def ensure_index(pinecone_client: Pinecone) -> None:
    """Create the Pinecone index if it does not yet exist."""
    # `list_indexes()` returns a list of objects, so extract the names
    existing_indexes = [idx.get("name") for idx in pinecone_client.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' already exists.")
        return

    print(f"Creating index '{INDEX_NAME}'...")
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until the index is ready
    print("Waiting for index to be ready...")
    while not pinecone_client.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(5)
    print("Index created and ready.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # --- 3. Point to the DB-build folder (dbjson) ---
    data_dir = base_dir / "dbjson"

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        return

    if load_dotenv is not None:
        # Load a .env file (e.g., token.env)
        # If the .env is not in the same directory, specify a path like load_dotenv("token.env")
        load_dotenv("token.env")

    # Check required environment variables
    require_env("OPENAI_API_KEY")
    pinecone_api_key = require_env("PINECONE_API_KEY")

    # Load all documents
    all_ids, all_documents = load_documents(data_dir)
    if not all_documents:
        print("No JSON documents found to upsert.")
        return

    total_docs = len(all_documents)
    print(f"Loaded {total_docs} documents into memory.")

    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Initialize Pinecone client and ensure the index exists
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    ensure_index(pinecone_client)

    # Initialize PineconeVectorStore
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # --- 4. Batch processing loop with rate-limit mitigation ---
    print(
        f"Upserting {total_docs} documents to index '{INDEX_NAME}' in batches of {BATCH_SIZE}..."
    )

    total_upserted = 0
    for i in range(0, total_docs, BATCH_SIZE):
        batch_docs = all_documents[i : i + BATCH_SIZE]
        batch_ids = all_ids[i : i + BATCH_SIZE]

        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (
            (total_docs // BATCH_SIZE) + 1
            if total_docs % BATCH_SIZE != 0
            else (total_docs // BATCH_SIZE)
        )

        print(
            f"\n--- Processing Batch {batch_num} / {total_batches} ({len(batch_docs)} documents) ---"
        )

        try:
            # LangChain `add_documents` performs embedding and upsert
            vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
            total_upserted += len(batch_docs)
            print(f"Batch {batch_num} upserted successfully.")

            # Wait between batches except after the final batch
            if i + BATCH_SIZE < total_docs:
                print(f"Waiting for {SLEEP_TIME} seconds to avoid rate limits...")
                time.sleep(SLEEP_TIME)

        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            print("Skipping this batch and continuing...")
            time.sleep(SLEEP_TIME)  # Wait on errors as well

    print(f"\n--- Complete ---")
    print(
        f"Total {total_upserted} / {total_docs} documents upserted into Pinecone index '{INDEX_NAME}'."
    )


if __name__ == "__main__":
    main()

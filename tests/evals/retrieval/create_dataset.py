import json
from pathlib import Path


def load_locomo(path):
    """Load LOCOMO data as JSON or JSONL."""
    text = Path(path).read_text(encoding="utf-8").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Support JSONL: one JSON record per line
        data = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    # Support common wrapper formats
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        elif "records" in data and isinstance(data["records"], list):
            data = data["records"]
        else:
            data = [data]

    return data


def normalize_question_type(question_type):
    """Map LOCOMO question types to broader benchmark categories."""

    mapping = {
        "temporal-reasoning": "temporal",
        "multi-hop": "multi-hop",
        "single-hop": "single-hop",
        "open-domain": "open-ended",
    }

    return mapping.get(
        str(question_type).lower(),
        question_type or "unknown"
    )


def convert_locomo(input_path, output_path):
    records = load_locomo(input_path)

    memories = []
    queries = []

    # Prevent duplicate memories across question records
    memory_lookup = {}

    memory_counter = 1

    for record in records:

        query_id = record.get("question_id")

        if not query_id:
            print("Skipping record without question_id")
            continue

        relevant_memory_ids = []

        sessions = record.get("haystack_sessions", [])
        session_ids = record.get("haystack_session_ids", [])

        # Process conversation sessions
        for session_index, session in enumerate(sessions):

            session_id = (
                session_ids[session_index]
                if session_index < len(session_ids)
                else f"{query_id}_session{session_index}"
            )

            for turn_index, message in enumerate(session):

                role = message.get("role", "unknown")
                content = message.get("content", "")

                if not content or not content.strip():
                    continue

                # Include speaker attribution in the memory text
                memory_content = f"{role}: {content.strip()}"

                # Identify a message consistently across records
                memory_key = (
                    session_id,
                    turn_index,
                    role,
                    content.strip()
                )

                # Reuse existing memory if already processed
                if memory_key in memory_lookup:
                    memory_id = memory_lookup[memory_key]

                else:
                    memory_id = f"m{memory_counter:06d}"
                    memory_counter += 1

                    memory_lookup[memory_key] = memory_id

                    memories.append({
                        "memory_id": memory_id,
                        "content": memory_content
                    })

                # Use LOCOMO's has_answer annotation as relevance label
                if message.get("has_answer", False):
                    relevant_memory_ids.append(memory_id)

        # Build query record
        query = {
            "query_id": query_id,
            "query": record.get("question", ""),
            "relevant_memory_ids": sorted(set(relevant_memory_ids)),
            "type": normalize_question_type(
                record.get("question_type")
            )
        }

        queries.append(query)

    # Final output
    output = {
        "description": (
            "Memory retrieval benchmark converted from LOCOMO. "
            "Memories are conversation messages. Each query lists "
            "the memory IDs annotated as relevant."
        ),
        "memories": memories,
        "queries": queries
    }

    Path(output_path).write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Conversion complete!")
    print(f"Memories: {len(memories)}")
    print(f"Queries: {len(queries)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    INPUT_FILE = "../longmemeval_oracle.json"
    OUTPUT_FILE = "dataset.json"

    convert_locomo(INPUT_FILE, OUTPUT_FILE)
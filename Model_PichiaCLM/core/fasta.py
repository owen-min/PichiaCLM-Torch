from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    id: str
    description: str
    sequence: str


def parse_fasta(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    current_header: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append(_record_from_parts(current_header, current_lines))
            current_header = line[1:].strip()
            current_lines = []
            continue
        if current_header is None:
            raise ValueError("FASTA content must start with a header line beginning with '>'.")
        current_lines.append(line)

    if current_header is not None:
        records.append(_record_from_parts(current_header, current_lines))

    if not records:
        raise ValueError("No FASTA records found.")
    return records


def parse_fasta_file(path: str | Path) -> list[FastaRecord]:
    return parse_fasta(Path(path).read_text(encoding="utf-8"))


def format_fasta(records: list[FastaRecord], line_width: int = 80) -> str:
    chunks = []
    for record in records:
        header = record.id
        if record.description:
            header = f"{header} {record.description}"
        chunks.append(f">{header}")
        sequence = "".join(record.sequence.split())
        chunks.extend(sequence[index : index + line_width] for index in range(0, len(sequence), line_width))
    return "\n".join(chunks) + "\n"


def _record_from_parts(header: str, sequence_lines: list[str]) -> FastaRecord:
    if not header:
        raise ValueError("FASTA header must not be empty.")
    sequence = "".join(sequence_lines).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"FASTA record '{header}' has an empty sequence.")
    parts = header.split(maxsplit=1)
    record_id = parts[0]
    description = parts[1] if len(parts) > 1 else ""
    return FastaRecord(id=record_id, description=description, sequence=sequence)

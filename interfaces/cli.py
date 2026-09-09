from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from Model_PichiaCLM.core.analysis import analyze_cds, load_training_codon_reference
from Model_PichiaCLM.core.biology import normalize_dna
from Model_PichiaCLM.core.config import DEFAULT_WEIGHTS_PATH
from Model_PichiaCLM.core.fasta import FastaRecord, format_fasta, parse_fasta_file
from Model_PichiaCLM.core.postprocess import conservative_postprocess
from Model_PichiaCLM.core.predictor import PichiaCLMPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PichiaCLM AA-to-CDS inference.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--aa", help="Amino acid sequence, e.g. MSTNPKPQR")
    input_group.add_argument("--aa-fasta", help="Input amino acid FASTA file for batch prediction.")
    input_group.add_argument("--cds", help="Externally optimized CDS to analyze without model prediction.")
    input_group.add_argument("--cds-fasta", help="Input CDS FASTA file to analyze without model prediction.")
    parser.add_argument(
        "--expected-aa",
        help="Expected amino acid sequence for translation consistency checks in CDS analysis mode.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to Arch1-0404.weights.pt",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cpu or cuda. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Allow X/Z/B/U/O ambiguous amino acids using the notebook UNK token.",
    )
    parser.add_argument("--analysis", action="store_true", help="Include sequence quality analysis.")
    parser.add_argument("--postprocess", action="store_true", help="Try conservative synonymous post-processing.")
    parser.add_argument(
        "--motif",
        action="append",
        default=[],
        help="Unwanted DNA motif to scan for. Can be passed more than once.",
    )
    parser.add_argument(
        "--restriction-site",
        action="append",
        default=[],
        help="Custom restriction site as Name=SEQUENCE or just SEQUENCE. Can be passed more than once.",
    )
    parser.add_argument("--out-fasta", help="Write optimized CDS sequences to this FASTA file.")
    parser.add_argument("--out-csv", help="Write a prediction summary table to this CSV file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cds or args.cds_fasta:
        _run_cds_analysis(args)
        return

    predictor = PichiaCLMPredictor(weights_path=args.weights, device=args.device)
    records = _input_records(args)
    payload_records = []
    fasta_records = []
    summary_rows = []

    training_reference, _ = load_training_codon_reference()
    for record in records:
        result = predictor.predict(record.sequence, allow_unknown=args.allow_unknown)
        payload = asdict(result)
        analysis = analyze_cds(
            result.cds,
            amino_acids=result.amino_acids,
            motifs=args.motif,
            custom_restriction_sites=args.restriction_site,
        )
        if args.analysis:
            payload["analysis"] = asdict(analysis)
        if args.postprocess:
            postprocess = conservative_postprocess(
                result.cds,
                result.amino_acids,
                reference_fractions=training_reference,
                forbidden_motifs=args.motif,
                custom_restriction_sites=args.restriction_site,
            )
            payload["postprocess"] = asdict(postprocess)
            output_cds = postprocess.optimized_cds
        else:
            output_cds = result.cds

        if args.aa_fasta:
            payload["id"] = record.id
            payload["description"] = record.description
        payload_records.append(payload)
        fasta_records.append(FastaRecord(id=record.id, description="PichiaCLM optimized CDS", sequence=output_cds))
        summary_rows.append(
            {
                "id": record.id,
                "aa_length": len(result.amino_acids),
                "cds_length": len(output_cds),
                "gc_percent": analysis.gc_percent,
                "cai_training": analysis.cai.training,
                "cai_public": analysis.cai.public,
                "translation_match": analysis.translation_matches_input,
                "restriction_sites": len(analysis.restriction_sites),
                "motif_hits": len(analysis.motif_hits),
                "local_gc_warnings": len(analysis.local_gc_outliers),
                "postprocess_replacements": len(payload.get("postprocess", {}).get("replacements", [])),
            }
        )

    if args.out_fasta:
        Path(args.out_fasta).write_text(format_fasta(fasta_records), encoding="utf-8")
    if args.out_csv:
        _write_csv(args.out_csv, summary_rows)

    response_payload = payload_records[0] if args.aa else {"records": payload_records}

    if args.json:
        print(json.dumps(response_payload, ensure_ascii=False))
        return

    for payload in payload_records:
        if "id" in payload:
            print(f"ID: {payload['id']}")
        print(f"Input AA: {payload['amino_acids']}")
        print(f"Device: {payload['device']}")
        print(f"Codon IDs: {payload['codon_ids']}")
        print(f"CDS: {payload['cds']}")
        if args.analysis:
            analysis_payload = payload["analysis"]
            print(f"GC%: {analysis_payload['gc_percent']} ({analysis_payload['gc_status']})")
            print(f"CAI training/public: {analysis_payload['cai']['training']} / {analysis_payload['cai']['public']}")
            print(f"Translation match: {analysis_payload['translation_matches_input']}")
            print(f"Restriction sites: {len(analysis_payload['restriction_sites'])}")
            print(f"Local GC warnings: {len(analysis_payload['local_gc_outliers'])}")
            print(f"Rare codon runs: {len(analysis_payload['rare_codon_runs'])}")
            print(f"Homopolymers: {len(analysis_payload['homopolymers'])}")
            print(f"Tandem repeats: {len(analysis_payload['tandem_repeats'])}")
            print(f"Repeated 12-mers: {len(analysis_payload['repeated_kmers'])}")
            print(f"Motif hits: {len(analysis_payload['motif_hits'])}")
        if args.postprocess:
            print(f"Postprocess replacements: {len(payload['postprocess']['replacements'])}")
        print()


def _input_records(args: argparse.Namespace) -> list[FastaRecord]:
    if args.aa:
        return [FastaRecord(id="sequence_1", description="", sequence=args.aa)]
    return parse_fasta_file(args.aa_fasta)


def _run_cds_analysis(args: argparse.Namespace) -> None:
    records = _cds_records(args)
    payload_records = []
    summary_rows = []
    for record in records:
        analysis = analyze_cds(
            record.sequence,
            amino_acids=args.expected_aa,
            motifs=args.motif,
            custom_restriction_sites=args.restriction_site,
        )
        payload = {
            "id": record.id,
            "description": record.description,
            "cds": normalize_dna(record.sequence),
            "expected_amino_acids": args.expected_aa,
            "translated_amino_acids": analysis.translated_amino_acids,
            "analysis": asdict(analysis),
        }
        payload_records.append(payload)
        summary_rows.append(
            {
                "id": record.id,
                "cds_length": analysis.cds_length,
                "codon_count": analysis.codon_count,
                "gc_percent": analysis.gc_percent,
                "cai_training": analysis.cai.training,
                "cai_public": analysis.cai.public,
                "translation_match": analysis.translation_matches_input,
                "restriction_sites": len(analysis.restriction_sites),
                "motif_hits": len(analysis.motif_hits),
                "local_gc_warnings": len(analysis.local_gc_outliers),
                "invalid_bases": ",".join(analysis.invalid_bases),
            }
        )

    if args.out_csv:
        _write_csv(args.out_csv, summary_rows)

    response_payload = payload_records[0] if args.cds else {"records": payload_records}
    if args.json:
        print(json.dumps(response_payload, ensure_ascii=False))
        return

    for payload in payload_records:
        if args.cds_fasta:
            print(f"ID: {payload['id']}")
        analysis_payload = payload["analysis"]
        print(f"CDS: {payload['cds']}")
        print(f"Translated AA: {payload['translated_amino_acids']}")
        print(f"Translation match: {analysis_payload['translation_matches_input']}")
        print(f"GC%: {analysis_payload['gc_percent']} ({analysis_payload['gc_status']})")
        print(f"Restriction sites: {len(analysis_payload['restriction_sites'])}")
        print(f"Local GC warnings: {len(analysis_payload['local_gc_outliers'])}")
        print(f"Motif hits: {len(analysis_payload['motif_hits'])}")
        print()


def _cds_records(args: argparse.Namespace) -> list[FastaRecord]:
    if args.cds:
        return [FastaRecord(id="cds_1", description="", sequence=args.cds)]
    return parse_fasta_file(args.cds_fasta)


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

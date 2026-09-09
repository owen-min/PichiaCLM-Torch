from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Model_PichiaCLM.core.analysis import analyze_cds, load_training_codon_reference
from Model_PichiaCLM.core.config import DEFAULT_WEIGHTS_PATH
from Model_PichiaCLM.core.postprocess import conservative_postprocess
from Model_PichiaCLM.core.predictor import PichiaCLMPredictor


class SequenceRecord(BaseModel):
    id: str = Field(..., examples=["seq1"])
    description: str = ""
    amino_acids: str = Field(..., examples=["MSTNPKPQR"])


class CdsRecord(BaseModel):
    id: str = Field(..., examples=["cds1"])
    description: str = ""
    cds: str = Field(..., examples=["ATGTCCACAAATCCCAAACCACAGAGA"])
    expected_amino_acids: str | None = Field(default=None, examples=["MSTNPKPQR"])


class PredictRequest(BaseModel):
    amino_acids: str = Field(..., examples=["MSTNPKPQR"])
    allow_unknown: bool = False
    include_analysis: bool = True
    unwanted_motifs: list[str] = Field(default_factory=list)
    custom_restriction_sites: list[str] = Field(default_factory=list)
    postprocess: bool = False


class PredictBatchRequest(BaseModel):
    records: list[SequenceRecord]
    allow_unknown: bool = False
    include_analysis: bool = True
    unwanted_motifs: list[str] = Field(default_factory=list)
    custom_restriction_sites: list[str] = Field(default_factory=list)
    postprocess: bool = False


class PredictResponse(BaseModel):
    id: str | None = None
    description: str | None = None
    amino_acids: str
    cds: str
    codon_ids: list[int]
    device: str
    analysis: dict[str, Any] | None = None
    postprocess: dict[str, Any] | None = None


class PredictBatchResponse(BaseModel):
    records: list[PredictResponse]


class AnalyzeCdsRequest(BaseModel):
    cds: str = Field(..., examples=["ATGTCCACAAATCCCAAACCACAGAGA"])
    expected_amino_acids: str | None = Field(default=None, examples=["MSTNPKPQR"])
    unwanted_motifs: list[str] = Field(default_factory=list)
    custom_restriction_sites: list[str] = Field(default_factory=list)


class AnalyzeCdsResponse(BaseModel):
    id: str | None = None
    description: str | None = None
    cds: str
    expected_amino_acids: str | None = None
    translated_amino_acids: str
    analysis: dict[str, Any]


class AnalyzeCdsBatchRequest(BaseModel):
    records: list[CdsRecord]
    unwanted_motifs: list[str] = Field(default_factory=list)
    custom_restriction_sites: list[str] = Field(default_factory=list)


class AnalyzeCdsBatchResponse(BaseModel):
    records: list[AnalyzeCdsResponse]


@lru_cache(maxsize=1)
def get_predictor() -> PichiaCLMPredictor:
    weights_path = os.environ.get("PICHIA_CLM_WEIGHTS", str(DEFAULT_WEIGHTS_PATH))
    device = os.environ.get("PICHIA_CLM_DEVICE") or None
    return PichiaCLMPredictor(weights_path=weights_path, device=device)


app = FastAPI(title="PichiaCLM Torch API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        payload = _predict_one(
            amino_acids=request.amino_acids,
            allow_unknown=request.allow_unknown,
            include_analysis=request.include_analysis,
            unwanted_motifs=request.unwanted_motifs,
            custom_restriction_sites=request.custom_restriction_sites,
            postprocess=request.postprocess,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> dict[str, object]:
    try:
        records = []
        for record in request.records:
            payload = _predict_one(
                amino_acids=record.amino_acids,
                allow_unknown=request.allow_unknown,
                include_analysis=request.include_analysis,
                unwanted_motifs=request.unwanted_motifs,
                custom_restriction_sites=request.custom_restriction_sites,
                postprocess=request.postprocess,
            )
            payload["id"] = record.id
            payload["description"] = record.description
            records.append(payload)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"records": records}


@app.post("/analyze_cds", response_model=AnalyzeCdsResponse)
def analyze_cds_endpoint(request: AnalyzeCdsRequest) -> dict[str, object]:
    try:
        return _analyze_external_cds(
            cds=request.cds,
            expected_amino_acids=request.expected_amino_acids,
            unwanted_motifs=request.unwanted_motifs,
            custom_restriction_sites=request.custom_restriction_sites,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze_cds_batch", response_model=AnalyzeCdsBatchResponse)
def analyze_cds_batch(request: AnalyzeCdsBatchRequest) -> dict[str, object]:
    try:
        records = []
        for record in request.records:
            payload = _analyze_external_cds(
                cds=record.cds,
                expected_amino_acids=record.expected_amino_acids,
                unwanted_motifs=request.unwanted_motifs,
                custom_restriction_sites=request.custom_restriction_sites,
            )
            payload["id"] = record.id
            payload["description"] = record.description
            records.append(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"records": records}


def _predict_one(
    amino_acids: str,
    allow_unknown: bool,
    include_analysis: bool,
    unwanted_motifs: list[str],
    custom_restriction_sites: list[str],
    postprocess: bool,
) -> dict[str, object]:
    result = get_predictor().predict(amino_acids, allow_unknown=allow_unknown)
    payload = asdict(result)
    analysis = analyze_cds(
        result.cds,
        amino_acids=result.amino_acids,
        motifs=unwanted_motifs,
        custom_restriction_sites=custom_restriction_sites,
    )
    if include_analysis:
        payload["analysis"] = asdict(analysis)
    if postprocess:
        training_reference, _ = load_training_codon_reference()
        payload["postprocess"] = asdict(
            conservative_postprocess(
                result.cds,
                result.amino_acids,
                reference_fractions=training_reference,
                forbidden_motifs=unwanted_motifs,
                custom_restriction_sites=custom_restriction_sites,
            )
        )
    return payload


def _analyze_external_cds(
    cds: str,
    expected_amino_acids: str | None,
    unwanted_motifs: list[str],
    custom_restriction_sites: list[str],
) -> dict[str, object]:
    analysis = analyze_cds(
        cds,
        amino_acids=expected_amino_acids,
        motifs=unwanted_motifs,
        custom_restriction_sites=custom_restriction_sites,
    )
    return {
        "cds": cds,
        "expected_amino_acids": expected_amino_acids,
        "translated_amino_acids": analysis.translated_amino_acids,
        "analysis": asdict(analysis),
    }

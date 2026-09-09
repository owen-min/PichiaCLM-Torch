# PichiaCLM Torch Deployment

This repository ships research notebooks plus trained PyTorch weights. The deployment code is split so the model logic stays independent from each user interface.

## Architecture

```text
Model_PichiaCLM/
  core/
    biology.py      # DNA/codon utilities and translation checks
    config.py       # default model paths
    fasta.py        # FASTA parsing and formatting
    fusion.py       # signal peptide + mature protein comparison
    model.py        # PyTorch model architecture
    postprocess.py  # conservative synonymous post-processing
    predictor.py    # model loading and AA-to-CDS prediction
    restriction.py  # restriction enzyme site scanning
    schemas.py      # shared result dataclasses
    vocab.py        # AA/CDS vocabularies and biological masks
  interfaces/
    cli.py          # command-line interface
    api.py          # FastAPI interface
    streamlit_app.py # Streamlit interface
```

`core` does not import FastAPI, Streamlit, or CLI code. CLI, API, and Streamlit are thin adapters over the same `PichiaCLMPredictor`.

Legacy module paths still work:

```powershell
python -m Model_PichiaCLM.cli --aa MSTNPKPQR
uvicorn Model_PichiaCLM.api:app --host 127.0.0.1 --port 8000
```

Prefer the new explicit paths:

```powershell
python -m Model_PichiaCLM.interfaces.cli --aa MSTNPKPQR
uvicorn Model_PichiaCLM.interfaces.api:app --host 127.0.0.1 --port 8000
streamlit run Model_PichiaCLM/interfaces/streamlit_app.py
```

## Install

Core inference only:

```powershell
pip install -r requirements-core.txt
```

FastAPI service:

```powershell
pip install -r requirements-api.txt
```

Streamlit UI:

```powershell
pip install -r requirements-streamlit.txt
```

Everything:

```powershell
pip install -r requirements-deploy.txt
```

## CLI

```powershell
python -m Model_PichiaCLM.interfaces.cli --aa MSTNPKPQR
python -m Model_PichiaCLM.interfaces.cli --aa MSTNPKPQR --json
python -m Model_PichiaCLM.interfaces.cli --aa MSTNPKPQR --analysis --json
python -m Model_PichiaCLM.interfaces.cli --aa-fasta input.fasta --analysis --out-fasta output_cds.fasta --out-csv report.csv
python -m Model_PichiaCLM.interfaces.cli --aa MSTNPKPQR --analysis --postprocess --restriction-site EcoRI=GAATTC
python -m Model_PichiaCLM.interfaces.cli --cds ATGTCCACAAATCCCAAACCACAGAGA --expected-aa MSTNPKPQR --analysis
python -m Model_PichiaCLM.interfaces.cli --cds-fasta optimized_cds.fasta --out-csv cds_qc_report.csv
```

## API

```powershell
uvicorn Model_PichiaCLM.interfaces.api:app --host 127.0.0.1 --port 8000
```

Call the API:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType application/json `
  -Body '{"amino_acids":"MSTNPKPQR"}'
```

Batch prediction:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict_batch `
  -ContentType application/json `
  -Body '{"records":[{"id":"seq1","amino_acids":"MSTNPKPQR"}],"include_analysis":true}'
```

Analyze externally optimized CDS without model prediction:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/analyze_cds `
  -ContentType application/json `
  -Body '{"cds":"ATGTCCACAAATCCCAAACCACAGAGA","expected_amino_acids":"MSTNPKPQR"}'
```

## Streamlit

Direct model mode:

```powershell
streamlit run Model_PichiaCLM/interfaces/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

FastAPI mode:

1. Start the API.
2. Start Streamlit.
3. Select `FastAPI` in the sidebar and point it at `http://127.0.0.1:8000`.

## Configuration

The default model weights path is:

```text
Model_PichiaCLM/Training/PichiaData/2Target_AllData/Arch1-0404.weights.pt
```

Override it for API mode:

```powershell
$env:PICHIA_CLM_WEIGHTS = "C:\path\to\Arch1-0404.weights.pt"
$env:PICHIA_CLM_DEVICE = "cpu"
uvicorn Model_PichiaCLM.interfaces.api:app --host 127.0.0.1 --port 8000
```

By default, ambiguous amino acids such as `X`, `Z`, `B`, `U`, and `O` are rejected because they do not have a biological codon mask. Use `--allow-unknown` in the CLI or `"allow_unknown": true` in the API only when you intentionally want the notebook's unknown-token behavior.

## Sequence analysis

Prediction responses can include a conservative quality report for the optimized CDS:

- translation consistency against the input amino acid sequence
- global GC content using a 35%-65% warning range
- local GC content using a 30 bp sliding window and a 25%-75% warning range
- CAI compared against both the repository training data and the public Kazusa `Pichia pastoris` codon table
- codon usage comparison, rare codon runs, homopolymers, tandem repeats, repeated 12-mers, and user-supplied unwanted motifs
- restriction enzyme site scanning for EcoRI, XhoI, NotI, BamHI, HindIII, NdeI, NcoI, KpnI, XbaI, and SpeI
- short/long amino acid sequence warnings
- optional conservative synonymous post-processing for restriction sites, unwanted motifs, homopolymers, high local GC, and repeated 12 bp fragments

The Streamlit UI provides Chinese tabs for single prediction, FASTA batch prediction, and signal peptide / mature protein fusion comparison.
It also includes a `二次优化 CDS 质检` tab for CDS sequences optimized outside this software; that path runs quality analysis only and does not call the prediction model.

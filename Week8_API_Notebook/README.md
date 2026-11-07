# Project 2: Chest X-Ray Pneumonia Screening API

A tested FastAPI service that serves my Week 6 computer-vision model and adds Week 7-style text evidence handling (PII masking, field extraction, urgent-term detection). It combines an X-ray image and a short clinical note into a single **REFER / ROUTINE** decision-support record.

---

## Contents
1. [What it does](#what-it-does)
2. [Project structure](#project-structure)
3. [Setup and run](#setup-and-run)
4. [API usage](#api-usage)
5. [Error handling and bad inputs](#error-handling-and-bad-inputs)
6. [Tests](#tests)
7. [Latency benchmark and capacity](#latency-benchmark-and-capacity)
8. [Model summary (Week 6)](#model-summary-week-6)
9. [Security and privacy](#security-and-privacy)
10. [Check Your Understanding](#check-your-understanding)
11. [Limitations](#limitations)

---

## What it does

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check: model loaded, class names, positive class |
| `/predict` | POST | Upload an X-ray (JPEG/PNG) and get label, class probabilities and latency |
| `/analyze` | POST | Upload an X-ray **and** a note; returns prediction, masked note, extracted fields, urgent terms, `REFER`/`ROUTINE` decision and reasons |

Design choices:

- **Model loaded once** at server start (`lifespan`), not per request.
- **Inference runs in a thread pool** (`run_in_threadpool`) so the async event loop stays free.
- **Preprocessing is rebuilt from the checkpoint** (`grayscale`, `img_size`, `backbone`, `classes`), so serving matches training.
- **PII is masked first** (CNIC and mobile number) before anything else touches the note.
- **Input validation:** content type, 5 MB size limit, readable image, note length limit of 2,000 characters.

## Project structure

```
p2-api/
├── app.py            # FastAPI service
├── test_api.py       # 8 automated tests (pytest)
├── bench.py          # latency benchmark against a running server
├── week6_model.pt    # Week 6 checkpoint (resnet18, 2 classes)
├── sample.jpeg       # one test X-ray used by tests and benchmark
├── docs/screenshots/ # proof of working calls and rejected inputs
├── requirements.txt
└── README.md
```

## Setup and run

Windows (Command Prompt). On Mac/Linux use `source .venv/bin/activate`.

```
mkdir p2-api
cd p2-api
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pillow pytest httpx numpy
```

Copy `week6_model.pt` and one test image (renamed `sample.jpeg`) into the folder, then start the server:

```
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive documentation ("Try it out").

**Configuration (environment variables):**

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `week6_model.pt` | Checkpoint file |
| `POSITIVE` | `PNEUMONIA` | The class that must not be missed |
| `THRESHOLD` | `0.5` | P(positive) at or above which the image triggers a referral in `/analyze` (see [operating point](#operating-point-note)) |
| `API_KEY` | *(unset)* | If set, every request needs an `X-API-Key` header |

## API usage

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "classes": ["NORMAL", "PNEUMONIA"], "positive": "PNEUMONIA"}
```

### `POST /predict`

```
curl -X POST http://127.0.0.1:8000/predict -F "file=@sample.jpeg;type=image/jpeg"
```

Response from my run (first call after server start, includes warm-up):

```json
{"label": "PNEUMONIA", "probabilities": {"NORMAL": 0.0074, "PNEUMONIA": 0.9926}, "latency_ms": 371.6}
```

`sample.jpeg` is one image from the *Chest X-Ray Images (Pneumonia)* dataset whose true class is PNEUMONIA, so the prediction above is correct for this image. A single image says nothing about overall accuracy; see the [model summary](#model-summary-week-6) for test-set results.

### `POST /analyze`

```
curl -X POST http://127.0.0.1:8000/analyze -F "file=@sample.jpeg;type=image/jpeg" -F "note=Cough and fever since Monday. Call 0300-1234567"
```

Response from my run:

```json
{
  "prediction": {"label": "PNEUMONIA", "probabilities": {"NORMAL": 0.0074, "PNEUMONIA": 0.9926}, "latency_ms": 174.4},
  "note_masked": "Cough and fever since Monday. Call [PHONE]",
  "fields": {},
  "urgent_terms": ["fever", "cough"],
  "decision": "REFER",
  "reasons": ["image: P(PNEUMONIA) = 0.99", "note mentions: fever, cough"],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

`fields` is empty here because the phone number was masked before field extraction. This is intended: raw identifiers are never returned.

**Decision rule:** `REFER` if P(`POSITIVE`) ≥ `THRESHOLD` **or** the note contains an urgent term; otherwise `ROUTINE`.

Screenshots of these calls are in [`docs/screenshots/`](docs/screenshots):

| Call | Screenshot |
|---|---|
| `/predict` success (200) | `docs/screenshots/predict_200.png` |
| `/analyze` success (200) | `docs/screenshots/analyze_200.png` |
| Rejected non-image (415) | `docs/screenshots/reject_415.png` |

## Error handling and bad inputs

Every error returns a short JSON message and no stack trace.

| Input | Status | Response |
|---|---|---|
| Non-image upload (`app.py`, `text/plain`) | **415** | `{"detail":"Upload a JPEG or PNG image"}` |
| Image larger than 5 MB (6,000,000-byte file) | **413** | `{"detail":"Image larger than 5 MB"}` |
| Corrupt file declared as `image/jpeg` | **400** | `{"detail":"File is not a readable image"}` |
| `/predict` with no file | **422** | FastAPI validation error (`file` field required) |
| Note longer than 2,000 characters | **422** | `{"detail":"Note longer than 2,000 characters"}` |
| Wrong or missing API key (when `API_KEY` is set) | **401** | `{"detail":"Invalid or missing API key"}` |

**Bad input tried beyond the automated tests:** I sent `/analyze` an image **without the `note` field**:

```
curl -s -w "\nHTTP status: %{http_code}\n" -X POST http://127.0.0.1:8000/analyze -F "file=@sample.jpeg;type=image/jpeg"
```

```
{"detail":[{"type":"missing","loc":["body","note"],"msg":"Field required","input":null}]}
HTTP status: 422
```

The API rejected the request with a clear validation error that names the missing field. It did not crash and it did not run the model. I also checked the 413 limit manually with a 6,000,000-byte file (`fsutil file createnew big.jpg 6000000`) and got `HTTP status: 413`.

## Tests

Eight automated tests in `test_api.py`:

| # | Test | What it proves |
|---|---|---|
| 1 | `test_health` | Service is up and the model is loaded |
| 2 | `test_predict_schema_and_probabilities` | 200 response and probabilities sum to 1 |
| 3 | `test_parity_with_direct_model` | API probabilities equal direct model calls (within 1e-3) |
| 4 | `test_rejects_wrong_type` | 415 for non-images |
| 5 | `test_rejects_too_large` | 413 above 5 MB |
| 6 | `test_rejects_corrupt_image` | 400 for unreadable images |
| 7 | `test_missing_file_is_422` | 422 when no file is sent |
| 8 | `test_analyze_masks_pii` | No part of the phone number or CNIC appears anywhere in the response, and `decision == REFER` |

Output of `pytest -q test_api.py`:

```
........                                                                 [100%]
================================ warnings summary ================================
.venv\Lib\site-packages\fastapi\testclient.py:1
  StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is
  deprecated; install `httpx2` instead.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
8 passed, 1 warning in 17.95s
```

The single warning is a library deprecation notice from Starlette/httpx. It does not affect the results.

## Latency benchmark and capacity

`bench.py` sends 50 sequential `POST /predict` requests to a running server.

**Hardware:** Intel Core i3-4010U @ 1.70 GHz (2 cores / 4 threads), 8 GB RAM, Windows, CPU-only PyTorch, one uvicorn worker.

```
n=50  p50=175 ms  p95=192 ms  max=387 ms
Throughput with 1 worker ~ 5.6 requests/s
```

| Metric | Value |
|---|---|
| p50 | 175 ms |
| p95 | 192 ms |
| max | 387 ms |
| Throughput (1 worker) | ≈ 5.6 requests/s |

p95 is close to p50, so latency is stable for typical requests. The 387 ms maximum is a single outlier; the first call after server start was also slower (371.6 ms in `/docs`), which is consistent with model warm-up, though I did not isolate the cause.

**Little's law (capacity of one worker).** L = λ × W. A single worker that is always busy has L = 1, and W = p50 = 0.175 s, so:

- λ = 1 / 0.175 ≈ **5.7 requests/s**
- Per day: 5.7 × 86,400 ≈ **494,000 requests/day** at 100% utilisation

This is an upper bound (the worker is assumed busy all the time) for sequential, CPU-only requests on this laptop. I did not test concurrent clients.

## Model summary (Week 6)

Full pipeline, audit and Grad-CAM analysis are in my Week 6 Kaggle notebook (*Week 6: Applied CV, Chest X-Ray Pneumonia*).

| Item | Value |
|---|---|
| Task | Binary image classification: NORMAL vs PNEUMONIA (chest X-ray), positive class = PNEUMONIA |
| Dataset | *Chest X-Ray Images (Pneumonia)*, Paul Mooney (Kaggle). Train folder: 5,216 images (NORMAL 1,341 / PNEUMONIA 3,875); official test folder: 624 images (NORMAL 234 / PNEUMONIA 390) |
| Image properties | Mostly grayscale (286 of 300 sampled; 14 RGB); sizes from 384 × 127 to 2510 × 2543, median 1300 × 890 |
| Imbalance ratio | 2.9 (PNEUMONIA : NORMAL) in the train folder |
| Split | `StratifiedGroupKFold` by patient id: train 4,354 / validation 862 / test 624 (official test folder). **0 patient groups shared between train and validation.** 1.83 images per patient group |
| Imbalance handling | Weighted cross-entropy, weight = N / (K × n_c) with N = 4,354, K = 2: NORMAL 4,354 / (2 × 1,125) = **1.935**, PNEUMONIA 4,354 / (2 × 3,229) = **0.674** |
| Duplicates | 62 exact-duplicate files in 30 groups (MD5, train and test combined), inspected and **not removed**. 0 of 624 test images had a near-duplicate in train (closest fingerprint distance 0.164 vs threshold 0.10) |
| Backbone and training | resnet18 (ImageNet weights). Phase 1: head only, 2 epochs. Phase 2: last block fine-tuned, 3 epochs. Best epoch chosen by **validation macro-F1** |
| Validation macro-F1 | 0.875 after phase 1, **0.964** after phase 2 (about 67 s per epoch on a Kaggle T4) |
| Test macro-F1 (argmax) | **0.878** (accuracy 0.893). Recall: NORMAL 0.726, PNEUMONIA 0.992. Weakest class: NORMAL |
| Threshold | **0.803**, chosen on validation for recall ≥ 0.95 |
| Test results at that threshold | Sensitivity **0.990** (386 of 390), specificity **0.795** (186 of 234), PPV 0.889 at the test prevalence of 62.5%, AUC 0.975 |
| PPV at a realistic prevalence | At 5% prevalence: 0.99 × 0.05 / (0.99 × 0.05 + 0.205 × 0.95) ≈ **0.203** |
| Grad-CAM verdict | **Shortcut suspected.** Some false positives were driven by regions outside the lungs, and one overlapped the letter marker on the image border. Not confirmed: a crop-and-re-score experiment was proposed but not run |

**Why PPV drops so much:** the test set is 62.5% pneumonia, but in a screening population with about 5% prevalence most positive calls would be false alarms (about 4 in 5). A positive result is a reason for a human review, not a diagnosis.

### Operating point note

`/predict` always returns the arg-max label. `THRESHOLD` only affects the `REFER` decision in `/analyze`, and its default is **0.5**, whereas the validated operating point from Week 6 is **0.803** (sensitivity 0.990, specificity 0.795). To serve the validated operating point:

```
set THRESHOLD=0.803
uvicorn app:app
```

(Mac/Linux: `THRESHOLD=0.803 uvicorn app:app`.) All benchmark and test results above were produced with the default 0.5.

## Security and privacy

- Optional API key via `API_KEY` (checked on every endpoint that accepts uploads). **Never commit a real key.**
- Uploads are limited to JPEG/PNG and 5 MB and are read in memory; nothing is written to disk.
- The note is masked for CNIC and mobile numbers **before** field extraction, and the raw note is never returned. A test fails if any part of the identifiers appears in the response.
- Use only de-identified images.

## Check Your Understanding

**1. A client uploads a PDF to `/predict`. Which status code should it get, and why not 500?**
**415 Unsupported Media Type.** The problem is the client's input, not a fault in the server, and 4xx codes tell the client exactly that: fix the request. A 500 means "the server broke", which is wrong here, it would hide the real cause, trigger false alarms in monitoring, and may leak a stack trace. Our code checks the content type first and returns a short message.

**2. Why is the model loaded in `lifespan` rather than inside the endpoint?**
Loading a checkpoint (≈45 MB) and rebuilding the network is slow compared to inference. If it happened inside the endpoint, every request would pay that cost on top of the ~175 ms of inference and would allocate memory again. Loading it once at startup keeps per-request latency low, shares one model across all requests, and lets the server fail fast at startup if the checkpoint is missing or wrong, while `/health` reports `model_loaded`.

**3. p50 = 40 ms and p95 = 400 ms. What does that tell you, and what would you check?**
Most requests are fast, but about 1 in 20 is roughly ten times slower: a **tail-latency** problem that the median hides. I would check (a) cold-start or warm-up requests, (b) queueing when several requests arrive together on one worker, (c) unusually large or slow-to-decode images, (d) garbage collection or other processes competing for the CPU. I would look at the whole latency distribution and per-request timings, separate cold from warm requests, and re-run with concurrent load.

**4. 500,000 requests a day, peak 5 times the average, 25 requests/s per worker. How many workers?**
Average = 500,000 / 86,400 ≈ 5.79 requests/s. Peak = 5 × 5.79 ≈ 28.9 requests/s. Workers = 28.9 / 25 ≈ 1.16, rounded up to **2 workers** (a third would add safety headroom).

**5. Why can model inference inside an `async def` endpoint slow down other requests?**
An `async def` endpoint runs on the single event loop. Model inference is CPU-bound and blocking, so while it runs the loop cannot serve anything else, even a cheap `/health` call, and other requests wait. Running inference with `run_in_threadpool` moves it to a worker thread and keeps the loop free (the lab's testing showed worst-case `/health` latency tripling without it).

**6. How many communication channels does a 4-person team have?**
n(n − 1) / 2 = 4 × 3 / 2 = **6** channels. This is why teams are kept to 2 or 3 people: a pair has 1 channel and a trio has 3.

## Limitations

- Trained and tested on one public dataset from a single source; it has **not** been validated on other hospitals, scanners, populations or paediatric/adult mixes.
- Train/test overlap check: 149 patient-group ids extracted from file names appear in both the training split and the official test folder. I did not verify whether these are the same patients or only reused id numbers, so the test score may be somewhat optimistic.
- Exact duplicates (62 files) were not removed from training, and the notebook does not record which splits they fall in.
- The test set (624 images, 62.5% pneumonia) is small and far more disease-heavy than a real screening population.
- Output is a probability from a small model, not a diagnosis; thresholds were chosen for high recall, which means false positives are expected.
- The text part is rule-based (regex and keyword matching), so it will miss spelling variants, Urdu or mixed-language notes, and unusual number formats.
- Benchmarks come from one laptop CPU with sequential requests; real concurrency and cloud hardware will differ.
- **Not for** clinical use, triage of real patients, or any decision without a qualified human reviewing the case.

# neural-ids — Roadmap

**Goal:** a learning and portfolio project — a neural network that flags suspicious network
flows. The repository ships the architecture and the code, not a trained model: anyone can
download it and train it on traffic they capture with Wireshark or tcpdump.

- **Phase 1 — Build (now):** the pipeline from capture files to model inputs, the two networks
  (from scratch in NumPy, then in PyTorch), and the commands to build datasets, train, and
  score — tested with synthetic data.
- **Phase 2 — Train (later):** train and evaluate on real captures, including public datasets.

The design (why flows, the networks, training modes, design decisions) is in
`docs/ARCHITECTURE.md`.

## Pipeline

```text
 capture.pcap / .pcapng ──► extract.py (T10): read packets, build flows ──┐
 (Wireshark, tcpdump)                                                     │
                                                                          ▼
 synthetic.py (T02): fake packet lists ─────────────► features.py (T02): one shared
 (testing and demo)                                   feature function
                                                                          │
                                                                          ▼
                      build-dataset (T11): one folder per label ──► flows table
                                                                          │
                                                                          ▼
                                                              preprocess.py (T03)
                                                                          │
                              ┌───────────────────────────────────────────┴──┐
                              ▼                                              ▼
                 classifier: needs labels                  autoencoder: normal traffic only
                 (T04–T06 NumPy, T08 PyTorch)              (T07 NumPy, T09 PyTorch)
                              └──────────────────────┬───────────────────────┘
                                                     ▼
                     model bundle + commands: train (T09), score and demo (T12)
```

## Status
Legend: `- [ ]` not started · `- [x]` done (with a link to the report)

### Phase 1 — Build
- [x] T00 — Project setup and workflow check · [spec](tasks/T00-setup.md) · [report](reports/T00-setup.md)
- [x] T01 — The feature contract: from packets to flows · [spec](tasks/T01-feature-contract.md) · [report](reports/T01-feature-contract.md)
- [ ] T02 — Flow features and synthetic traffic · [spec](tasks/T02-features-synthetic.md)
- [ ] T03 — Preprocessing without leakage · [spec](tasks/T03-preprocessing.md)
- [ ] T04 — Baselines: majority class, perceptron, logistic regression · [spec](tasks/T04-baselines-logreg.md)
- [ ] T05 — MLP from scratch, part 1: layers and backpropagation · [spec](tasks/T05-mlp-backprop.md)
- [ ] T06 — MLP from scratch, part 2: training loop and debugging · [spec](tasks/T06-mlp-training.md)
- [ ] T07 — Autoencoder from scratch: anomaly detection without labels · [spec](tasks/T07-autoencoder.md)
- [ ] T08 — PyTorch classifier and cross-check · [spec](tasks/T08-pytorch-classifier.md)
- [ ] T09 — PyTorch autoencoder, model bundle, and train command · [spec](tasks/T09-pytorch-autoencoder-bundle.md)
- [ ] T10 — From capture files to flows · [spec](tasks/T10-pcap-extractor.md)
- [ ] T11 — Bring your own data: labeled datasets from captures · [spec](tasks/T11-bring-your-own-data.md)
- [ ] T12 — End to end, demo, and portfolio docs · [spec](tasks/T12-end-to-end-portfolio.md)

### Phase 2 — Train (specs written when phase 1 is done)
- Public captures, processed with our own extractor (never with a dataset's precomputed
  features). Candidates checked on 2026-09-26:
  - UNSW-NB15 (2015): about 100 GB of pcaps captured with tcpdump, with a ground-truth file.
  - CIC-IDS2017 (2017): five days of pcaps with labeled attacks; known labeling errors.
  - CTU-13 (2011): real botnet traffic; only the botnet part of each capture is public.
  - MAWI archive: daily backbone traces with payloads removed and IP addresses anonymized;
    research use only; no attack labels (useful as real-world background traffic).
- Labeling rules for mixed captures (time windows and IP addresses, or the dataset's
  ground-truth file), and a faster packet reader for 100 GB-scale datasets.
- A lab dataset: normal traffic plus attacks you run between your own isolated VMs.
- Training and SOC-style evaluation: a false-positive budget, recall per attack type, a test on
  a different network or dataset, and regularization.
- Extensions: host-window features (to catch scans and brute force across many flows),
  sequence models over the first packets of each flow, and adversarial robustness.

## Map to the learning path

| Learning step | Tasks |
|---|---|
| 1. Python and NumPy essentials | T00–T03 |
| 2. Linear and logistic regression, gradient descent | T04 |
| 3. A single neuron and a perceptron | T04 |
| 4. MLP with backpropagation, from scratch | T05–T07 |
| 5. PyTorch fundamentals | T08–T09 |
| 6. Evaluation, overfitting, regularization | T04, T07, T08, T11; in depth in phase 2 with real data |
| 7. Small project | T10–T12, then phase 2 |

## Ground rules (details in CLAUDE.md)
- One feature function for every data source; identifiers are never model inputs.
- Training split for fitting, validation split for tuning, test split once; split by capture
  when the data comes from captures.
- Synthetic results are labeled as synthetic; every number comes from a real command.
- Every task ends with a report in `docs/reports/`, then a commit after review.

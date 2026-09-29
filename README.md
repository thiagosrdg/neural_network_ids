# neural-ids

A learning and portfolio project: a neural network that flags suspicious network flows.

A **flow** is a group of packets that share the same source and destination addresses,
ports, and protocol, like one conversation in Wireshark's *Statistics → Conversations*.
The model looks at statistics of each flow (sizes, timing, counts), never at payloads or
addresses.

The repository ships the architecture and the code, **not a trained model**. Anyone can
train it on traffic they capture with Wireshark or tcpdump (`.pcap` / `.pcapng`).

## Objectives

- Understand how neural networks work by building them **from scratch in NumPy**, then in
  **PyTorch**.
- Build a clean pipeline: capture file → flows → features → model → score.
- Compare two approaches: a **classifier** (learns from labeled traffic) and an
  **autoencoder** (learns only normal traffic and flags what looks different).
- Follow good ML and security practice: no data leakage, no identifiers as inputs, and no
  capture files in the repository.

## Phases

1. **Build (now):** pipeline and networks, tested with synthetic data. Results on synthetic
   data prove that the code works, not that detection works.
2. **Train (later):** train and evaluate on real captures and public datasets.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install Python 3.13 (some uv setups download
Python only on request), then all dependencies:

```bash
uv python install 3.13
uv sync
```

## Run the tests

```bash
uv run pytest -q
uv run ruff check .
```

## Plan and status

See [docs/ROADMAP.md](docs/ROADMAP.md). The design is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE). You may use, copy, and modify the code, as long as you keep the
copyright notice.

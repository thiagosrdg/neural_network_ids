# neural-ids — Setup and workflow guide

This guide gets the project running in VS Code with Claude Code on Fedora, explains every
Claude Code file in the kit, and lists current best practices with the reason for each.

**What you are building:** a learning and portfolio project — a neural network that flags
suspicious network flows. The repository ships the architecture and the code, not a trained
model: anyone can download it and train it on traffic they capture with Wireshark or tcpdump.
Phase 1 (now) builds the whole pipeline — capture file → flows → features → network → score —
from scratch in NumPy and then in PyTorch, tested with synthetic traffic. Phase 2 (later) trains
it on real captures, including public datasets. The design is in `docs/ARCHITECTURE.md`; the
plan is in `docs/ROADMAP.md`.

**How the kit works:** you do not write prompts. Each task has a stable spec in `docs/tasks/`,
so the prompts stay short: `/start-task T03` plus notes that Claude gives you in the claude.ai
Project after reviewing your last report.

---

## 1. The loop

```text
 claude.ai Project "Neural Network"              Claude Code in VS Code
 ──────────────────────────────────              ──────────────────────
 Claude reviews your report and    ── prompt ──►  /start-task T0x
 writes the next prompt                            1. concept + pseudocode + plan
                                                   2. you approve
                                                   3. tests + scaffolding
                                                   4. you write the TODO(human) pieces
                                                   5. verify (tests, lint, reviewer)
 paste the report here             ◄── report ──   6. report in docs/reports/ + commit
```

One task = one Claude Code conversation. Start a new conversation for every task.

---

## 2. One-time setup (Fedora)

**Account.** Claude Code needs a Pro, Max, Team, Enterprise, or Console account. The free plan
does not include it.

**VS Code.** Version 1.94 or newer. Use the RPM from Microsoft's repository
(https://code.visualstudio.com/docs/setup/linux), not the Flatpak: the Flatpak sandbox hides
host tools such as `uv` and `claude` from the integrated terminal.

**Claude Code CLI.** Recommended. The VS Code extension has its own copy for the chat panel,
but the CLI gives you `claude` in the terminal, the full command set, and `claude doctor`.

```bash
curl -fsSL https://claude.ai/install.sh | bash   # native installer, updates itself
claude --version
claude doctor
```

Prefer signed packages through dnf? Claude Code also has a signed RPM repository:

```bash
sudo tee /etc/yum.repos.d/claude-code.repo <<'EOF'
[claude-code]
name=Claude Code
baseurl=https://downloads.claude.ai/claude-code/rpm/stable
enabled=1
gpgcheck=1
gpgkey=https://downloads.claude.ai/keys/claude-code.asc
EOF
sudo dnf install claude-code
```

When dnf asks, check that the key fingerprint is
`31DD DE24 DDFA B679 F42D 7BD2 BAA9 29FF 1A7E CACE`. dnf installs do not update themselves;
run `sudo dnf upgrade claude-code` from time to time.

**uv.** A fast Python package and project manager. It also downloads Python 3.13 for this
project without touching the system Python.

```bash
sudo dnf install uv       # or the official installer: curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

**Wireshark command-line tools.** `dumpcap` captures traffic, `capinfos` describes a capture
file, and `tshark` analyzes one. Joining the `wireshark` group lets you capture without `sudo`:

```bash
sudo dnf install wireshark-cli
sudo usermod -aG wireshark "$USER"   # then log out and log back in
```

**Optional: sandbox tools** (see section 7):

```bash
sudo dnf install bubblewrap socat
```

---

## 3. Create the project from the kit

```bash
mkdir -p ~/projects && cd ~/projects
unzip ~/Downloads/neural-ids-starter-kit-v3.zip     # creates ./neural-ids
cd neural-ids
git init -b main
git add -A
git commit -m "chore: add Claude Code starter kit"
code .
```

The kit has hidden folders (`.claude/`, `.vscode/`). In the file manager, press `Ctrl+H` to see
them. The first commit gives you a clean baseline, so every change Claude makes is visible in
`git diff`.

In VS Code:
1. Trust the workspace when asked. Claude Code applies the project's allow rules only after you
   trust the folder.
2. Install the recommended extensions when prompted: Claude Code, Python, Jupyter, Ruff.
3. Open Claude Code: click the Spark icon (top right of an open file, or in the Activity Bar)
   and sign in.
4. Type `/` in the prompt box. You should see `/start-task`, `/check-my-code`, and
   `/task-report`. T00 also starts by listing the `CLAUDE.md` rules Claude can see. (In the
   CLI, `/context` shows `CLAUDE.md` under "Memory files".)
5. Choose a permission mode with the mode indicator under the prompt box. Suggestion: **Manual**
   for T00 and T01, so you see every action and learn what Claude Code does; then **Auto** or
   **Edit automatically** once you trust the workflow.

---

## 4. Captures: rules and your first capture

You need a capture of your own traffic for T01 (to study in Wireshark) and for T12 (the
end-to-end test). Training on real captures comes in phase 2.

**Rules**
- Capture only traffic you are allowed to capture: your own devices, your own lab, or a network
  whose owner gave you permission. Capturing other people's communications without permission
  can be a crime.
- Captures contain personal data: IP addresses, the domains you visit (DNS), and sometimes
  passwords sent without encryption. Keep them in `data/raw/`, which is git-ignored, and do not
  share them.
- Claude Code sends everything it reads to the model. The kit blocks Claude from reading
  `data/raw/` directly; the project's own code processes captures on your machine and prints
  only summaries.

**A 2-minute capture of your own computer**

With `dumpcap` (after the group setup in section 2, no `sudo` needed):

```bash
dumpcap -i any -a duration:120 -w data/raw/my-laptop.pcapng
capinfos data/raw/my-laptop.pcapng     # file type, link type, packets, duration
```

With `tcpdump`:

```bash
sudo timeout 120 tcpdump -i any -Z "$USER" -w data/raw/my-laptop.pcap
```

`-Z "$USER"` makes tcpdump write the file as your user; without it, Fedora's tcpdump switches
to its own `tcpdump` user and may not be allowed to write into your project folder.

Or in the Wireshark window: start a capture on your main interface, stop it after about two
minutes, then File → Save As (pcapng format).

---

## 5. Working on a task

1. Open a **new conversation** in the Claude Code panel.
2. Paste the prompt from the Project chat. It starts with `/start-task T0x`.
3. Read the plan (concept, pseudocode, file plan, your pieces, verification) and answer "OK",
   or correct it. For a big or unclear task, switch to **Plan** mode first (mode indicator, or
   type `/plan`).
4. When Claude stops at your `TODO(human)` pieces, write the code in the editor. The tests tell
   you when you are done.
5. At the end, answer the learning-check questions, review the report, and approve the commit.
6. Paste the report into the Project chat. You get feedback and the next prompt.

**Tutoring commands** (defined in `CLAUDE.md`):

| You type | What happens |
|---|---|
| `done` | Claude runs the tests for your piece and reviews your code. It explains problems but does not fix them. |
| `hint` | The next hint level: 1 = concept nudge, 2 = pseudocode, 3 = partial code with blanks. |
| `solution` | The full solution for the current piece, explained line by line. |
| `skip` | Claude writes the current piece (the report says so). |
| `quiz` | Three short questions about the current concept, one at a time. |
| `/check-my-code` | Tests, lint, and format check, plus a review of your code, with no edits. |
| `/task-report` | Rewrites the report, for example if a conversation ended early. |

To change how much code you write, edit the `Learning mode` line in `CLAUDE.md` (`hands-on`,
`guided`, or `fast`).

---

## 6. The Claude Code files in this kit

Claude Code only requires `CLAUDE.md`. The other files follow current best practice: keep the
always-loaded instructions short, and load procedures and specs only when they are needed.

| File | What it does | When Claude reads it |
|---|---|---|
| `CLAUDE.md` | Project rules: language, teaching style, learning mode, commands, data-pipeline, ML, security, and git rules | Every conversation, so it stays under 200 lines |
| `.claude/settings.json` | Shared settings: permission rules (allow / ask / deny), output style, environment | Every conversation |
| `.claude/settings.local.json` | Your personal overrides. Claude Code creates it when you choose "Yes, and don't ask again". Git-ignored. | Every conversation |
| `.claude/skills/start-task/SKILL.md` | `/start-task <ID>`: the standard workflow (plan → build → verify → report) | Only when you type the command |
| `.claude/skills/check-my-code/SKILL.md` | `/check-my-code`: quality gate and review, without edits | Only when you type the command |
| `.claude/skills/task-report/SKILL.md` | `/task-report`: writes the report from the template | Only when you type the command |
| `.claude/agents/ml-reviewer.md` | A subagent (a helper with its own clean context) that reviews diffs for leakage, contract violations, shortcut features, privacy problems, and invented numbers | When a task asks for a review |
| `docs/ARCHITECTURE.md` | The design: components, flows, the two networks, training modes, design decisions, limitations. Also the portfolio's main technical document. | Read by `/start-task` when a task touches the design |
| `docs/ROADMAP.md` | Plan, pipeline, and status checklist | Read by `/start-task` |
| `docs/tasks/*.md` | One spec per task: goal, deliverables, your pieces, acceptance criteria, learning check | Read by `/start-task` |
| `docs/reports/` | `TEMPLATE.md`, plus one report per task | Written at the end of each task |
| `.vscode/` | Editor settings and recommended extensions | Read by VS Code |

T01 adds the most important project file: the feature contract (`docs/features.md` and
`src/neural_ids/schema.py`), which defines exactly what the network sees.

What you do **not** need:
- `AGENTS.md`. Claude Code reads it only when there is no `CLAUDE.md`. If you later use another
  coding agent in this repo too, move the shared rules into `AGENTS.md` and make `CLAUDE.md`
  start with the line `@AGENTS.md`, followed by the Claude-only rules.
- `@` imports of big files in `CLAUDE.md`. Imported files load into every conversation and use
  context. Point to files instead, as the kit does with `docs/`.
- `/init`. The kit already has a `CLAUDE.md`. Running `/init` later only suggests improvements.
- `CLAUDE.local.md` is optional, for private notes that should not be committed (it is
  git-ignored).

Tip: HTML comments (`<!-- ... -->`) in `CLAUDE.md` are removed before Claude reads the file, so
you can leave notes for yourself at no cost.

---

## 7. Best practices (and why)

**Context is the scarce resource.** Claude's context window holds the whole conversation:
every message, file, and command output. Quality drops as it fills.
- Use a new conversation for every task (or `/clear` in the CLI). Old, unrelated history
  distracts Claude.
- `/compact <focus>` summarizes the conversation on demand. In the CLI, `/context` shows what
  fills the window.
- `/btw <question>` asks a side question that does not enter the conversation history.
- After two failed corrections on the same problem, stop. Start a new conversation with a
  better prompt that includes what you learned. (Tell me in the Project; I will write it.)

**Plan before building.** `/start-task` always shows a plan first. Plan mode is worth it when a
change touches several files or you are unsure of the approach; skip it for tiny edits.

**Give Claude a way to check its own work.** Tests are the specification, and Claude must show
real command output. "Show me the command and its output" is always a fair request. A
fresh-context review catches what the author misses: the `ml-reviewer` subagent does this for
ML and pipeline mistakes, and the bundled `/code-review` skill reviews a diff for bugs.

**Correct course early.** Interrupt Claude as soon as it goes off track (`Esc` in the CLI, the
stop control in the VS Code panel). In VS Code, hover over a message and use the rewind button
to restore the code, the conversation, or both to that point (CLI: `Esc` `Esc` or `/rewind`).
Checkpoints do not track changes made by shell commands, so git is your real safety net: commit
at the end of every task.

**Keep instructions lean.** For each line in `CLAUDE.md`, ask: "Would removing this cause a
mistake?" If Claude keeps ignoring a rule, the file is probably too long. Procedures belong in
skills; hard guarantees belong in permission rules, tests, or hooks.

**Defense in depth (security and privacy).**
- Permissions: the kit allows only safe, repetitive commands (tests, lint, project modules,
  `capinfos`); always asks before `git commit`, `git push`, `tshark`, `tcpdump`, and `dumpcap`;
  and denies `sudo`, reads of `.env` files, and any direct read or edit of `data/raw/`. See all
  rules with `/permissions` (VS Code: command menu → Customize → Permissions).
- Personal data stays out of the context: Claude works with summaries and anonymized tables,
  never with raw captures.
- No accidental traffic: the rules forbid Scapy's sending functions, and T10 adds tests that
  fail if anything tries to send a packet.
- Prompt injection: text inside files, captures, or web pages can contain instructions.
  `CLAUDE.md` tells Claude to treat such text as data, but still read what Claude runs,
  especially in Auto mode.
- Sandbox (optional): operating-system-level isolation for shell commands, using bubblewrap on
  Linux. Enable it after T00 (when Python is already installed): install `bubblewrap` and
  `socat`, run `/sandbox`, and pick a mode. uv and matplotlib write outside the project folder,
  so add this to `.claude/settings.local.json` (merge it with anything already there):

  ```json
  {
    "sandbox": {
      "enabled": true,
      "filesystem": {
        "allowWrite": ["~/.cache/uv", "~/.local/share/uv", "~/.cache/matplotlib"]
      },
      "network": {
        "allowedDomains": ["pypi.org", "files.pythonhosted.org", "download.pytorch.org"]
      }
    }
  }
  ```

  If a command fails inside the sandbox, Claude asks to retry it outside (the prompt says
  "unsandboxed"). Mention it in the report and we will adjust the rules.

**Use the editor integration.**
- Select code and press `Alt+K` to insert a reference like `@src/neural_ids/features.py#10-25`.
- `@terminal:<name>` shares a terminal's output without copy and paste.
- Plans open as Markdown documents where you can add comments before Claude starts.
- Session history (top of the panel) resumes old conversations. In the terminal, use
  `claude --continue` or `claude --resume`.

**Grow the setup slowly.** Add a rule to `CLAUDE.md` only when Claude repeats a mistake. A good
next step after T03: ask Claude to write a hook that runs `ruff format` after every edit (hooks
run every time; instructions are only advice).

**Output style.** The kit sets the built-in **Explanatory** style, which adds short "Insight"
notes about the choices behind the code. Longer answers use more of your usage limits; switch
with `/output-style concise` in the CLI, or the command menu → Output styles in VS Code.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `/start-task` does not appear | Check the path `.claude/skills/start-task/SKILL.md`; run "Developer: Reload Window"; run `claude doctor`. |
| Claude ignores `CLAUDE.md` | Is the file at the repo root? In the CLI, `/context` lists it under "Memory files". |
| "Settings Error" at start | Settings files are strict JSON: no comments, no trailing commas. `claude doctor` shows the problem. |
| Allow rules seem ignored | Trust the workspace; project allow rules wait for trust. |
| `uv` or `claude` not found in the VS Code terminal | You are probably on the Flatpak build of VS Code; use the RPM. |
| `dumpcap` says you lack permission | Check `groups` lists `wireshark`; log out and back in after `usermod`. |
| `tcpdump` says `Permission denied` for the output file | Add `-Z "$USER"`, or capture with `dumpcap`. |
| The extractor cannot read a capture | Run `capinfos <file>` and paste its output (not the file) into the chat; the link type may be unsupported. |
| Plots do not open | By design: scripts save figures to `reports/figures/`. The kit sets `MPLBACKEND=Agg` for the commands Claude runs. |

---

## 9. References

- Claude Code best practices: https://code.claude.com/docs/en/best-practices
- CLAUDE.md and memory: https://code.claude.com/docs/en/memory
- Skills: https://code.claude.com/docs/en/skills
- Settings: https://code.claude.com/docs/en/settings
- Permissions: https://code.claude.com/docs/en/permissions
- Sandboxing: https://code.claude.com/docs/en/sandboxing
- VS Code extension: https://code.claude.com/docs/en/vs-code
- Output styles: https://code.claude.com/docs/en/output-styles
- uv and PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/
- Scapy: https://scapy.readthedocs.io/
- dumpcap: https://www.wireshark.org/docs/man-pages/dumpcap.html
- tcpdump: https://www.tcpdump.org/manpages/tcpdump.1.html
- Engelen, Rimmer, Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017
  Case Study" (WTMC 2021): https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf

# Cool Code

**What if your coding assistant had a hive mind?**

Cool Code is a swarm-powered CLI coding agent where multiple AI agents race, vote, and collaborate to give you the best answer. Built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) with Claude + MiniMax (M2.5/M2.7) support.

```
  ____            _    ____          _
 / ___|___   ___ | |  / ___|___   __| | ___
| |   / _ \ / _ \| | | |   / _ \ / _` |/ _ \
| |__| (_) | (_) | | | |__| (_) | (_| |  __/
 \____\___/ \___/|_|  \____\___/ \__,_|\___|

  Swarm-powered coding agent | v0.1.0
```

## Why Cool Code?

| Capability | Traditional Agents | Cool Code |
|---|---|---|
| Agent Collaboration | Agents work in isolation | Queen-led swarms with shared memory and consensus |
| Decision Making | Single agent decides | 5 consensus algorithms (Raft, Byzantine, Gossip, Weighted, Majority) |
| LLM Providers | Single provider | Claude + MiniMax racing in parallel — best answer wins |
| Memory | Session-only | HNSW vector memory + PageRank knowledge graph + SQLite persistence |
| Task Routing | Manual | Learned pattern matching that improves over time |
| Planning | Manual breakdown | Automatic decomposition across 5 domains |
| Specialized Workflows | One-size-fits-all | 6 goal pipelines (security audit, code review, debug, ...) |
| Interactive Control | Wait or kill | ESC to cancel, type mid-execution to inject context |
| Cost Visibility | Silent token burn | Real-time cost display, per-task tracking, budget caps |
| Autonomous Pipelines | One step at a time | Multi-stage workflows with human gates & git checkpoints |
| Background Intelligence | Reactive only | Daemon mode watches files, detects issues proactively |

## Quick Start

### Install

```bash
pip install git+https://github.com/ankitkr3/CoolCode.git
```

### Run

```bash
coolcode
```

On first launch, Cool Code walks you through setup interactively:

```
╭─────────────────────────────────────────────╮
│  Welcome to Cool Code!                      │
│                                             │
│  Let's set up your LLM provider(s).         │
│  This only happens once — your config is    │
│  saved to ~/.coolcode/config.json           │
│  and persists across sessions.              │
╰─────────────────────────────────────────────╯

Step 1: Which provider(s) do you want to use?

  1  Claude (Anthropic)        — Best code quality
  2  MiniMax (M2.5 / M2.7)    — Fast & affordable
  3  Both (parallel racing)    — Best of both worlds

Choose (1/2/3): 3

Step 2: Select a claude model:

  1  Claude Sonnet 4.6          (Balanced)
  2  Claude Opus 4.6            (Highest quality)
  3  Claude Haiku 4.5           (Fastest)

Choose (1-3) [default: 1]: 1

Enter claude API key: sk-ant-...

Step 2: Select a minimax model:

  1  MiniMax M2.7 highspeed     (Fastest)
  2  MiniMax M2.7               (Best quality)
  3  MiniMax M2.5               (Best value)
  4  MiniMax M2.5 highspeed     (Fast & cheap)

Choose (1-4) [default: 1]: 1

Enter minimax API key: ...

✓ Setup complete! Config saved to ~/.coolcode/config.json
```

Your config persists across sessions. Use `/model` anytime to switch providers or models.

You can also set API keys via environment variables (these override saved config):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export MINIMAX_API_KEY=...
```

### One-shot mode

```bash
coolcode "fix the login bug in auth.py"
coolcode -s quality "refactor the payment module"
coolcode -w 5 -c byzantine "implement user dashboard"
```

## How It Works

```
User Task
    │
    ▼
┌──────────────────┐
│  Queen Delegator  │  ← Analyzes task, picks worker types
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│        Parallel Worker Spawning       │
│                                      │
│  ┌─────────────┐  ┌─────────────┐   │
│  │ coder-claude │  │coder-minimax│   │  ← Same task, different LLMs
│  └─────────────┘  └─────────────┘   │
│  ┌─────────────┐  ┌─────────────┐   │
│  │debug-claude  │  │debug-minimax│   │  ← Racing in parallel
│  └─────────────┘  └─────────────┘   │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────┐
│  Queen Evaluator  │  ← Picks winner via consensus
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Queen Coordinator │  ← Merges results if needed
└────────┬─────────┘
         │
         ▼
    Best Answer + Stats
```

### Goal Pipeline Flow

When a goal is active (`/goal cyber-security`, `/goal code-review`, etc.), the queen is bypassed entirely. Instead, a predetermined pipeline of specialized workers runs in stages:

```
User Task + Active Goal
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 1: Research                   │
│  ┌────────────────────────────────┐  │
│  │ researcher — Attack Surface    │  │  ← Shared file-read cache
│  └────────────────────────────────┘  │    across all workers
└────────────────┬─────────────────────┘
                 │ results feed into ▼
┌──────────────────────────────────────┐
│  Stage 2: Parallel Analysis          │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │Injection │ │Auth/Crypto│ │Supply│ │  ← 3 specialists in parallel
│  │ Analysis │ │ Analysis  │ │Chain │ │    (no provider racing)
│  └──────────┘ └──────────┘ └──────┘ │
└────────────────┬─────────────────────┘
                 │ results feed into ▼
┌──────────────────────────────────────┐
│  Stage 3: Consolidation              │
│  ┌────────────────────────────────┐  │
│  │ reviewer — Deduplicate & Score │  │
│  └────────────────────────────────┘  │
└────────────────┬─────────────────────┘
                 │ results feed into ▼
┌──────────────────────────────────────┐
│  Stage 4: Remediation Roadmap        │
│  ┌────────────────────────────────┐  │
│  │ planner — Prioritized Fixes    │  │
│  └────────────────────────────────┘  │
└────────────────┬─────────────────────┘
                 ▼
    Final Report + Stats
```

## Architecture

```
coolcode/
├── cli.py                  # Interactive REPL + one-shot CLI
├── config.py               # Auto-detects providers from env vars
├── learner.py              # WorkflowLearner — learns what works over time
├── agent/
│   ├── swarm.py            # Hive mind orchestrator (with cancellation + file cache)
│   ├── queen.py            # 3 queen types (Coordinator, Evaluator, Delegator)
│   ├── worker.py           # 8 worker types (Coder, Reviewer, Debugger, ...)
│   ├── consensus.py        # 5 consensus algorithms
│   ├── router.py           # Learned task routing
│   └── goals.py            # 6 goal pipelines (code-review, cyber-security, ...)
├── memory/
│   ├── vector_store.py     # HNSW vector memory (sub-ms search)
│   ├── knowledge_graph.py  # PageRank + community detection
│   ├── collective.py       # Shared memory (LRU cache + SQLite)
│   ├── scoped.py           # 3-scope memory (project/local/user)
│   └── learning_bridge.py  # Connects all memory subsystems
├── llm/
│   └── provider.py         # Claude + MiniMax with failover & cost routing
├── tools/
│   ├── files.py            # read, write, edit, list, glob
│   ├── tracked.py          # Tracked tools with shared file-read cache
│   ├── shell.py            # Shell execution with safety checks
│   ├── search.py           # Grep + definition finder
│   └── git.py              # Git operations
├── auto/
│   ├── pipeline.py         # Autonomous multi-stage pipeline orchestration
│   ├── gates.py            # Human approval gates (approve/reject/edit/rollback)
│   └── checkpoints.py      # Git checkpoint management for rollback
├── daemon/
│   ├── server.py           # Background daemon with Unix socket IPC
│   ├── watchers.py         # File and git event watchers
│   ├── insights.py         # Proactive insight engine (heuristic-based)
│   └── ipc.py              # IPC client for interactive CLI
├── cost.py                 # Cost tracking, budgets, daily persistence
├── planning/
│   ├── decomposer.py       # Auto task decomposition (5 domains)
│   └── parallel.py         # Parallel agent racing
└── prompts/
    └── system.py           # 4-phase reasoning protocol
```

## Features

### Swarm Intelligence
- **3 Queen Types**: Coordinator (merges results), Evaluator (picks best), Delegator (routes tasks)
- **8 Worker Types**: Coder, Reviewer, Planner, Researcher, Debugger, Tester, Refactorer, Security
- **Parallel Racing**: Same task runs on Claude AND MiniMax simultaneously — best answer wins

### 5 Consensus Algorithms
- **Majority**: Simple >50% vote
- **Weighted**: Votes scaled by agent confidence
- **Raft**: Leader-based consensus with term elections
- **Byzantine**: Fault-tolerant (tolerates f < n/3 faulty agents)
- **Gossip**: Epidemic-style convergence

### 6 Goal Pipelines

Goals are predetermined multi-stage workflows that bypass generic queen routing for domain-specific tasks. Each goal runs a pipeline of specialized workers with shared file caching.

| Goal | Description | Stages |
|---|---|---|
| `/goal code-review` | Deep code review — correctness, security, performance, style | 3 (scan → review → remediation plan) |
| `/goal cyber-security` | World-class security audit — OWASP, CWE, SANS, supply chain | 4 (attack surface → 3 parallel analyses → consolidation → roadmap) |
| `/goal build-feature` | Full feature lifecycle — plan, implement, test, review | 4 |
| `/goal debug` | Scientific debugging — observe, hypothesize, test, fix | 3 |
| `/goal explain` | Architecture explanation — structure, data flow, design decisions | 2 |
| `/goal optimize` | Performance optimization — profile, identify, fix bottlenecks | 3 |
| `/goal general` | Default mode — queen routes everything | 1 |

Goal pipelines include:
- **Shared file-read cache** — workers in the same pipeline don't re-read the same files
- **No provider racing** — goals already parallelize via multiple workers per stage
- **Stage-level context passing** — each stage sees results from all previous stages
- **Immediate ESC cancellation** — pressing ESC cancels all running workers, not just between stages

### Interactive Controls

- **ESC** — cancel current execution immediately (all workers stop gracefully)
- **Type + Enter** — inject additional context mid-execution without stopping

### Cost Visibility

Real-time cost tracking across all providers — no more silent token burn.

- **Live cost in activity panel**: see `$0.0023 | 3.2s` as workers execute
- **Per-task cost**: every task shows input/output tokens and USD cost in stats
- **Budget caps**: `/budget daily 5.00` or `/budget session 2.00`
- **Auto-warning**: alerts at 80% budget, blocks at 100%
- **Provider breakdown**: see which provider costs what in `/stats`
- **Daily persistence**: costs logged to `~/.coolcode/costs.jsonl`

### Autonomous Pipelines (`/auto`)

Multi-stage autonomous workflows with human approval gates and git rollback.

```
Cool Code > /auto build user dashboard with auth and tests

┌──────────────────────────────────────┐
│  Autonomous Pipeline Plan            │
├────┬─────────────────┬───────┬──────┤
│  # │ Stage           │ Gate  │ Cost │
├────┼─────────────────┼───────┼──────┤
│  1 │ Plan & Arch     │ GATE  │ $0.01│
│  2 │ Implement Core  │       │ $0.02│
│  3 │ Write Tests     │       │ $0.01│
│  4 │ Code Review     │ GATE  │ $0.02│
│  5 │ Apply Fixes     │       │ $0.01│
└────┴─────────────────┴───────┴──────┘

Start pipeline? (y/n): y
```

At each **GATE**, execution pauses and you choose:
- **Approve** — continue to next stage
- **Reject** — stop the pipeline
- **Edit** — give corrective instructions and re-run the stage
- **Rollback** — reset to before this stage (via git checkpoint)

### Daemon Mode (`/daemon`)

Background process that watches your project and proactively surfaces insights.

```bash
Cool Code > /daemon start
Daemon started watching: /Users/you/project
```

The daemon:
- **Watches file changes** — detects rapid edits, growing files, missing test updates
- **Watches git events** — branch switches, new commits
- **Surfaces insights** between prompts: *"auth.py changed 5 times but tests not updated"*
- **Learns patterns** — which files are edited together, common workflows
- **Zero cost** — uses heuristics, no LLM calls

```
Cool Code >
  ! [daemon] auth.py changed 5 times but tests not updated
  > [daemon] cli.py is 750 lines — consider splitting it
```

### Multi-Provider LLM
- **Claude** (Sonnet, Opus, Haiku) — best code quality
- **MiniMax** (M2.7, M2.7-highspeed, M2.5, M2.5-highspeed) — fast and affordable
- **Automatic failover**: if one provider fails, the other takes over
- **Cost-based routing**: MiniMax for cheap tasks, Claude for quality-critical ones
- **Live switching**: `/model claude`, `/model minimax`, `/model both`

### Memory System
- **HNSW Vector Store**: Sub-millisecond semantic search over code and conversations
- **Knowledge Graph**: PageRank identifies the most influential code entities; community detection clusters related knowledge
- **Collective Memory**: SQLite-backed shared memory with LRU cache, 8 memory types (fact, pattern, insight, decision, error, context, preference, learning)
- **3-Scope Memory**: Project-level, local machine, and global user preferences — with cross-agent transfer

### Intelligent Task Routing
- Learns from past routing decisions
- Pattern-matches new tasks against historical outcomes
- Automatically improves accuracy over time

### Better Prompting
4-phase reasoning protocol: **Understand → Plan → Execute → Verify**
- Anti-hallucination guardrails
- Context-aware behavior per language/project type
- Self-correction loop instructions

## CLI Commands

| Command | Description |
|---|---|
| `/model` | Show active providers and models |
| `/model setup` | Re-run provider setup (change keys, models) |
| `/model claude` | Switch to Claude only |
| `/model minimax` | Switch to MiniMax only |
| `/model both` | Enable parallel racing (both providers) |
| `/model claude:claude-opus-4-6` | Switch Claude to a specific model |
| `/goal` | Show current goal + available goals |
| `/goal code-review` | Deep code review mode |
| `/goal cyber-security` | Security audit (OWASP/CWE/SANS) |
| `/goal build-feature` | Plan → implement → test → review |
| `/goal debug` | Diagnose → fix → verify |
| `/goal explain` | Architecture explanation |
| `/goal optimize` | Performance optimization |
| `/goal general` | Back to default (queen routes everything) |
| `/auto <task>` | Autonomous pipeline with gates & checkpoints |
| `/cost` | Quick cost summary (session + today) |
| `/budget` | View/set budget caps |
| `/budget daily 5.00` | Set daily budget to $5 |
| `/budget session 2` | Set session budget to $2 |
| `/budget off` | Remove budget limits |
| `/daemon start` | Start background file/git watcher |
| `/daemon stop` | Stop the daemon |
| `/daemon status` | Show daemon info and stats |
| `/stats` | Provider stats, learnings, cost, and memory |
| `/help` | Show all commands |
| `quit` | Exit Cool Code |

All `/model` changes are **persisted** to `~/.coolcode/config.json` — they survive restarts.

## CLI Options

```
coolcode [TASK] [OPTIONS]

Options:
  -p, --provider [claude|minimax]   Preferred LLM provider
  -s, --strategy [cost|quality|fast] Routing strategy (default: cost)
  -w, --workers INTEGER              Number of parallel workers (default: 3)
  -c, --consensus [majority|weighted|raft|byzantine|gossip]
                                     Consensus algorithm (default: weighted)
  -v, --verbose                      Enable debug logging
```

## Supported Models

| Provider | Model | Tier | Cost (per 1M tokens) |
|---|---|---|---|
| Claude | claude-sonnet-4-6 | Balanced | $3 in / $15 out |
| Claude | claude-opus-4-6 | Highest quality | $15 in / $75 out |
| Claude | claude-haiku-4-5 | Fastest | $0.80 in / $4 out |
| MiniMax | MiniMax-M2.7-highspeed | Fastest | $0.50 in / $2.00 out |
| MiniMax | MiniMax-M2.7 | Best quality | $1.00 in / $4.00 out |
| MiniMax | MiniMax-M2.5 | Best value | $0.15 in / $1.10 out |
| MiniMax | MiniMax-M2.5-highspeed | Fast & cheap | $0.30 in / $2.20 out |

## Versioning

Cool Code uses semantic versioning with git tags. You can install a specific version:

```bash
# Latest
pip install git+https://github.com/ankitkr3/CoolCode.git

# Specific version (rollback)
pip install git+https://github.com/ankitkr3/CoolCode.git@v0.1.0
```

## Development

```bash
git clone https://github.com/ankitkr3/CoolCode.git
cd CoolCode
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT

# Cool Code

**What if your coding assistant had a hive mind?**

Cool Code is a swarm-powered CLI coding agent where multiple AI agents race, vote, and collaborate to give you the best answer. Built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) with Claude + MiniMax M2.5 support.

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
  2  MiniMax M2.5              — Cheapest, very fast
  3  Both (parallel racing)    — Best of both worlds

Choose (1/2/3): 3

Step 2: Select a claude model:

  1  Claude Sonnet 4.6          (Balanced)
  2  Claude Opus 4.6            (Highest quality)
  3  Claude Haiku 4.5           (Fastest)

Choose (1-3) [default: 1]: 1

Enter claude API key: sk-ant-...

Step 2: Select a minimax model:

  1  MiniMax M2.5               (Best value)
  2  MiniMax M2.5 Lightning     (Fastest)

Choose (1-2) [default: 1]: 1

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

## Architecture

```
coolcode/
├── cli.py                  # Interactive REPL + one-shot CLI
├── config.py               # Auto-detects providers from env vars
├── agent/
│   ├── swarm.py            # Hive mind orchestrator
│   ├── queen.py            # 3 queen types (Coordinator, Evaluator, Delegator)
│   ├── worker.py           # 8 worker types (Coder, Reviewer, Debugger, ...)
│   ├── consensus.py        # 5 consensus algorithms
│   └── router.py           # Learned task routing
├── memory/
│   ├── vector_store.py     # HNSW vector memory (sub-ms search)
│   ├── knowledge_graph.py  # PageRank + community detection
│   ├── collective.py       # Shared memory (LRU cache + SQLite)
│   └── scoped.py           # 3-scope memory (project/local/user)
├── llm/
│   └── provider.py         # Claude + MiniMax with failover & cost routing
├── tools/
│   ├── files.py            # read, write, edit, list, glob
│   ├── shell.py            # Shell execution with safety checks
│   ├── search.py           # Grep + definition finder
│   └── git.py              # Git operations
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

### Multi-Provider LLM
- **Claude** (Sonnet, Opus, Haiku) — best code quality
- **MiniMax M2.5** — 80.2% SWE-Bench, 10-20x cheaper than Opus
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
| `/stats` | Show provider performance stats |
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

| Provider | Model | Speed | Cost (per 1M tokens) |
|---|---|---|---|
| Claude | claude-sonnet-4-6 | Balanced | $3 in / $15 out |
| Claude | claude-opus-4-6 | Highest quality | $15 in / $75 out |
| Claude | claude-haiku-4-5 | Fastest | $0.80 in / $4 out |
| MiniMax | MiniMax-M2.5 | 50 tok/s | $0.15 in / $1.20 out |
| MiniMax | MiniMax-M2.5-Lightning | 100 tok/s | $0.30 in / $2.40 out |

## Development

```bash
git clone https://github.com/ankitkr3/CoolCode.git
cd CoolCode
pip install -e ".[dev]"
```

## License

MIT

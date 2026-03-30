"""Goal system — specialized workflows that replace generic queen routing.

Each goal defines a predetermined pipeline of workers with domain-specific prompts.
When a goal is active, the queen delegator is bypassed entirely — workers execute
in the exact order defined by the pipeline.

Usage:
    /goal                    — show current goal + available goals
    /goal code-review        — switch to deep code review mode
    /goal general            — back to default (queen routes everything)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coolcode.agent.worker import WorkerType


@dataclass
class GoalStep:
    """A single step in a goal pipeline."""

    worker_type: WorkerType
    prompt_override: str  # replaces default WORKER_PROMPTS for this step
    label: str  # human-readable step name, e.g. "OWASP Analysis"
    timeout: int = 0  # 0 = use per-worker-type timeout from WORKER_TIMEOUTS


@dataclass
class GoalStage:
    """A stage in the pipeline. Steps within a stage run in parallel."""

    steps: list[GoalStep]
    merge_strategy: str = "concatenate"  # "concatenate" | "consensus"


@dataclass
class GoalPipeline:
    """Complete pipeline for a goal."""

    name: str
    description: str
    stages: list[GoalStage]
    output_format: str = ""  # instructions for final output formatting


# =============================================================================
# GOAL: code-review
# =============================================================================

_CODE_REVIEW = GoalPipeline(
    name="code-review",
    description="Deep code review — correctness, security, performance, style",
    stages=[
        # Stage 1: Research the codebase
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.RESEARCHER,
                label="Codebase Scan",
                timeout=120,
                prompt_override=(
                    "You are the RESEARCH phase of a deep code review.\n\n"
                    "Your job:\n"
                    "1. Scan the codebase structure — list_dir, glob_search for key files\n"
                    "2. Read the files the user wants reviewed (or the most critical files if unspecified)\n"
                    "3. Map: imports, dependencies, call graphs, data flow\n"
                    "4. Identify the tech stack, frameworks, patterns used\n\n"
                    "Output a STRUCTURED MAP:\n"
                    "- Files examined (with line counts)\n"
                    "- Key classes/functions and their roles\n"
                    "- Data flow: input → processing → output\n"
                    "- External dependencies and API boundaries\n"
                    "- Areas of high complexity or risk\n\n"
                    "Be fast. Read at most 10-12 key files. Use grep_search for patterns."
                ),
            ),
        ]),

        # Stage 2: Deep review
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.REVIEWER,
                label="Deep Code Review",
                timeout=0,
                prompt_override=(
                    "You are a SENIOR CODE REVIEWER performing a comprehensive review.\n\n"
                    "You have the codebase map from Stage 1. Now do a DEEP review.\n\n"
                    "CHECK EACH CATEGORY:\n\n"
                    "## CORRECTNESS (Critical)\n"
                    "- Logic errors, off-by-one, wrong comparisons\n"
                    "- Null/undefined handling, missing error checks\n"
                    "- Race conditions, concurrency issues\n"
                    "- API contract violations (wrong types, missing fields)\n"
                    "- Resource leaks (unclosed files, connections, streams)\n"
                    "- Exception handling: swallowed errors, wrong exception types\n\n"
                    "## SECURITY (Critical)\n"
                    "- Injection: SQL, XSS, command, path traversal\n"
                    "- Auth: broken authentication, missing authorization checks\n"
                    "- Data exposure: secrets in code, PII in logs, insecure transmission\n"
                    "- Input validation: missing sanitization at trust boundaries\n\n"
                    "## PERFORMANCE (High)\n"
                    "- N+1 queries, unnecessary loops, O(n^2) where O(n) possible\n"
                    "- Missing indexes, uncached repeated computation\n"
                    "- Memory: large allocations, unbounded growth, unnecessary copies\n"
                    "- I/O: sync where async needed, blocking calls\n\n"
                    "## DESIGN (Medium)\n"
                    "- SOLID violations, tight coupling, god objects\n"
                    "- DRY violations, copy-paste code\n"
                    "- Naming: unclear, misleading, inconsistent\n"
                    "- Abstraction: too much or too little\n\n"
                    "## STYLE (Low)\n"
                    "- Formatting, convention violations\n"
                    "- Dead code, unused imports\n"
                    "- Missing/excessive comments\n\n"
                    "OUTPUT FORMAT — for each finding:\n"
                    "```\n"
                    "### [SEVERITY] Title\n"
                    "**File:** path/to/file.py:line\n"
                    "**Category:** Correctness|Security|Performance|Design|Style\n"
                    "**Issue:** What's wrong\n"
                    "**Impact:** What could go wrong\n"
                    "**Fix:** How to fix it (with code if helpful)\n"
                    "```\n\n"
                    "Sort findings by severity: CRITICAL > HIGH > MEDIUM > LOW.\n"
                    "Be specific. Reference exact lines. No vague suggestions."
                ),
            ),
        ]),

        # Stage 3: Remediation plan
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.PLANNER,
                label="Remediation Plan",
                timeout=90,
                prompt_override=(
                    "You are creating a REMEDIATION PLAN from code review findings.\n\n"
                    "You have the review findings from Stage 2.\n\n"
                    "Create an actionable plan:\n\n"
                    "## Executive Summary\n"
                    "- Total findings by severity\n"
                    "- Overall code health assessment (1-10)\n"
                    "- Top 3 most critical issues\n\n"
                    "## Remediation Roadmap\n"
                    "### P0 — Fix Immediately (Critical/Security)\n"
                    "### P1 — Fix This Sprint (High)\n"
                    "### P2 — Fix Next Sprint (Medium)\n"
                    "### P3 — Backlog (Low/Style)\n\n"
                    "For each item: specific file, specific change, estimated effort.\n"
                    "Group related fixes that should be done together.\n"
                    "Do NOT repeat the full findings — reference them by title."
                ),
            ),
        ]),
    ],
    output_format=(
        "Combine all stages into a final report:\n"
        "# Code Review Report\n"
        "## Executive Summary (from Stage 3)\n"
        "## Findings (from Stage 2, sorted by severity)\n"
        "## Remediation Roadmap (from Stage 3)"
    ),
)


# =============================================================================
# GOAL: cyber-security
# =============================================================================

_CYBER_SECURITY = GoalPipeline(
    name="cyber-security",
    description="World-class security audit — OWASP, CWE, SANS, supply chain",
    stages=[
        # Stage 1: Attack surface mapping
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.RESEARCHER,
                label="Attack Surface Mapping",
                timeout=0,
                prompt_override=(
                    "You are a SECURITY RESEARCHER mapping the attack surface.\n\n"
                    "Systematically identify ALL entry points and trust boundaries:\n\n"
                    "## Entry Points\n"
                    "- HTTP/API endpoints (routes, methods, params, headers)\n"
                    "- CLI arguments and stdin\n"
                    "- File inputs (uploads, config files, data files)\n"
                    "- Environment variables used in logic\n"
                    "- Database queries and ORM usage\n"
                    "- IPC, message queues, WebSocket handlers\n"
                    "- Cron jobs, scheduled tasks, background workers\n\n"
                    "## Data Flow\n"
                    "- Trace user input from entry → processing → storage → output\n"
                    "- Identify where sanitization/validation happens (or doesn't)\n"
                    "- Map trust boundaries (user input vs internal data)\n\n"
                    "## Authentication & Authorization\n"
                    "- Auth mechanisms used (JWT, sessions, API keys, OAuth)\n"
                    "- Authorization checks (RBAC, ACL, middleware guards)\n"
                    "- Which endpoints lack auth checks\n\n"
                    "## Dependencies\n"
                    "- List all third-party packages (from requirements.txt, package.json, etc.)\n"
                    "- Note any that handle security-critical operations\n"
                    "- Flag obviously outdated or unmaintained deps\n\n"
                    "## Cryptography\n"
                    "- Encryption/hashing used and where\n"
                    "- Key/secret management approach\n"
                    "- TLS/SSL configuration\n\n"
                    "Output a STRUCTURED INVENTORY. Read the actual code. Do NOT guess."
                ),
            ),
        ]),

        # Stage 2: Parallel security analysis (3 workers, different domains)
        GoalStage(
            steps=[
                GoalStep(
                    worker_type=WorkerType.SECURITY,
                    label="Injection & Input Validation (OWASP A03)",
                    timeout=0,
                    prompt_override=(
                        "You are a SECURITY ENGINEER specializing in INJECTION VULNERABILITIES.\n\n"
                        "Using the attack surface map from Stage 1, analyze for:\n\n"
                        "## SQL Injection (CWE-89)\n"
                        "- Raw SQL with string formatting/concatenation\n"
                        "- ORM raw queries, extra() calls\n"
                        "- Stored procedures with unsanitized params\n\n"
                        "## Cross-Site Scripting — XSS (CWE-79)\n"
                        "- Reflected: user input echoed in responses\n"
                        "- Stored: user content rendered without escaping\n"
                        "- DOM-based: client-side JS using unsanitized data\n"
                        "- Template injection in Jinja2/Mako/etc.\n\n"
                        "## Command Injection (CWE-78)\n"
                        "- os.system(), subprocess with shell=True\n"
                        "- eval(), exec(), __import__() with user data\n"
                        "- Template string formatting used in shell commands\n\n"
                        "## Path Traversal (CWE-22)\n"
                        "- File operations using unsanitized user paths\n"
                        "- Directory listing without containment\n"
                        "- Zip extraction without path validation (zip slip)\n\n"
                        "## SSRF (CWE-918)\n"
                        "- User-controlled URLs in HTTP requests\n"
                        "- DNS rebinding opportunities\n"
                        "- Internal service access via URL manipulation\n\n"
                        "## XML/XXE (CWE-611)\n"
                        "- XML parsing with external entities enabled\n"
                        "- SOAP endpoints without entity restrictions\n\n"
                        "## LDAP Injection (CWE-90)\n"
                        "- LDAP queries with unsanitized input\n\n"
                        "## Header Injection (CWE-113)\n"
                        "- User input in HTTP response headers\n"
                        "- CRLF injection opportunities\n\n"
                        "FOR EACH FINDING:\n"
                        "```\n"
                        "### [CVSS Score] Title\n"
                        "**CWE:** CWE-XXX\n"
                        "**OWASP:** A0X:YYYY\n"
                        "**File:** path:line\n"
                        "**Vulnerability:** Exact description\n"
                        "**Proof of Concept:** How to exploit\n"
                        "**Remediation:** Specific fix with code\n"
                        "```\n\n"
                        "Only report CONFIRMED findings in the actual code. No hypotheticals."
                    ),
                ),
                GoalStep(
                    worker_type=WorkerType.SECURITY,
                    label="Auth, Crypto & Data Exposure (OWASP A01/A02/A04/A07)",
                    timeout=0,
                    prompt_override=(
                        "You are a SECURITY ENGINEER specializing in AUTH, CRYPTO & DATA PROTECTION.\n\n"
                        "Using the attack surface map from Stage 1, analyze for:\n\n"
                        "## Broken Access Control — OWASP A01 (CWE-862, CWE-863)\n"
                        "- Missing authorization checks on endpoints\n"
                        "- IDOR: direct object references without ownership checks\n"
                        "- Privilege escalation: user accessing admin functions\n"
                        "- CORS misconfiguration allowing cross-origin access\n"
                        "- Missing CSRF protection on state-changing operations\n"
                        "- JWT issues: no expiry, weak signing, missing validation\n\n"
                        "## Cryptographic Failures — OWASP A02 (CWE-327, CWE-328, CWE-330)\n"
                        "- Weak algorithms: MD5/SHA1 for passwords, DES/RC4\n"
                        "- Hardcoded keys/salts/IVs\n"
                        "- ECB mode, missing HMAC, no key rotation\n"
                        "- Passwords: plaintext storage, weak hashing (no bcrypt/argon2)\n"
                        "- PRNG: using random instead of secrets/os.urandom\n"
                        "- Missing TLS, allowing HTTP, mixed content\n\n"
                        "## Broken Authentication — OWASP A07 (CWE-287, CWE-384)\n"
                        "- Credential stuffing: no rate limiting on login\n"
                        "- Weak password policy\n"
                        "- Session: predictable IDs, no timeout, no invalidation on logout\n"
                        "- Missing MFA on sensitive operations\n"
                        "- Password reset: insecure token generation/validation\n\n"
                        "## Sensitive Data Exposure (CWE-200, CWE-312, CWE-319)\n"
                        "- PII in logs (emails, IPs, passwords, tokens)\n"
                        "- Sensitive data in error messages/stack traces\n"
                        "- API responses leaking internal data\n"
                        "- Data at rest without encryption\n"
                        "- Caching sensitive responses (Cache-Control headers)\n\n"
                        "## Insecure Design — OWASP A04\n"
                        "- Missing rate limiting on critical operations\n"
                        "- No account lockout\n"
                        "- Business logic flaws (race conditions in payments, etc.)\n"
                        "- Missing audit trail for sensitive operations\n\n"
                        "FOR EACH FINDING — same format as the injection analysis:\n"
                        "Include CWE ID, OWASP category, CVSS 3.1 score, exact location, PoC, remediation code."
                    ),
                ),
                GoalStep(
                    worker_type=WorkerType.SECURITY,
                    label="Supply Chain, Secrets & Config (OWASP A05/A06/A08/A09)",
                    timeout=0,
                    prompt_override=(
                        "You are a SECURITY ENGINEER specializing in SUPPLY CHAIN & CONFIGURATION.\n\n"
                        "Using the attack surface map from Stage 1, analyze for:\n\n"
                        "## Hardcoded Secrets (CWE-798)\n"
                        "- API keys, passwords, tokens in source code\n"
                        "- Private keys, certificates in repository\n"
                        "- Database connection strings with credentials\n"
                        "- AWS/GCP/Azure credentials\n"
                        "- .env files committed to repo\n"
                        "- Secrets in CI/CD configs, Dockerfiles, docker-compose\n\n"
                        "## Security Misconfiguration — OWASP A05 (CWE-16)\n"
                        "- Debug mode enabled in production configs\n"
                        "- Default credentials not changed\n"
                        "- Unnecessary features/services enabled\n"
                        "- Missing security headers (CSP, HSTS, X-Frame-Options, etc.)\n"
                        "- Permissive CORS (Access-Control-Allow-Origin: *)\n"
                        "- Directory listing enabled\n"
                        "- Stack traces/error details exposed to users\n"
                        "- Insecure cookie flags (missing HttpOnly, Secure, SameSite)\n\n"
                        "## Vulnerable Dependencies — OWASP A06\n"
                        "- Known CVEs in dependencies (check versions in requirements/package files)\n"
                        "- Unpinned dependencies (version ranges allowing vulnerable versions)\n"
                        "- Outdated packages with known issues\n"
                        "- Dependency confusion risk (internal package names)\n"
                        "- Typosquatting risk on package names\n"
                        "- Sub-dependency risks\n\n"
                        "## Software Integrity — OWASP A08 (CWE-502)\n"
                        "- Insecure deserialization (pickle, yaml.load, JSON with eval)\n"
                        "- Missing integrity checks on downloads/updates\n"
                        "- CI/CD pipeline security\n"
                        "- Unsigned packages/artifacts\n\n"
                        "## Logging & Monitoring — OWASP A09 (CWE-778)\n"
                        "- Missing audit logs for auth events\n"
                        "- Sensitive data in logs (CWE-532)\n"
                        "- No alerting on suspicious activity\n"
                        "- Log injection vulnerabilities\n"
                        "- Missing request IDs for traceability\n\n"
                        "## OWASP A10 — Server-Side Request Forgery (SSRF)\n"
                        "- Already covered in injection analysis, but cross-check\n\n"
                        "FOR EACH FINDING — same format as other security analyses."
                    ),
                ),
            ],
            merge_strategy="concatenate",
        ),

        # Stage 3: Consolidation and scoring
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.REVIEWER,
                label="Vulnerability Consolidation",
                timeout=120,
                prompt_override=(
                    "You are a SECURITY LEAD consolidating vulnerability findings.\n\n"
                    "You have findings from 3 parallel security analyses (Stage 2).\n\n"
                    "Your job:\n"
                    "1. DEDUPLICATE — remove duplicates found by multiple analysts\n"
                    "2. CROSS-REFERENCE — identify findings that compound each other\n"
                    "   (e.g., XSS + missing CSP = higher severity)\n"
                    "3. VALIDATE — remove false positives or hypothetical findings\n"
                    "4. SCORE — assign final CVSS 3.1 scores\n"
                    "5. MAP — map each finding to:\n"
                    "   - OWASP Top 10 (2021)\n"
                    "   - CWE Top 25 (2023)\n"
                    "   - SANS Top 25\n\n"
                    "Output a UNIFIED VULNERABILITY REPORT:\n\n"
                    "## Vulnerability Summary Table\n"
                    "| # | Title | CVSS | OWASP | CWE | File | Priority |\n"
                    "| - | ----- | ---- | ----- | --- | ---- | -------- |\n\n"
                    "## Detailed Findings\n"
                    "(grouped by OWASP category, sorted by CVSS within each group)\n\n"
                    "## Compliance Mapping\n"
                    "- OWASP Top 10 coverage\n"
                    "- CWE Top 25 coverage\n"
                    "- Items checked and found clean"
                ),
            ),
        ]),

        # Stage 4: Remediation roadmap
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.PLANNER,
                label="Remediation Roadmap",
                timeout=90,
                prompt_override=(
                    "You are creating a SECURITY REMEDIATION ROADMAP.\n\n"
                    "From the consolidated vulnerability report (Stage 3), create:\n\n"
                    "## Executive Summary\n"
                    "- Total vulnerabilities by severity\n"
                    "- Security posture rating (A-F)\n"
                    "- Top 3 most critical risks\n"
                    "- Estimated total remediation effort\n\n"
                    "## P0 — Fix Immediately (CVSS >= 9.0, actively exploitable)\n"
                    "For each: exact code change, file, estimated time\n\n"
                    "## P1 — Fix Within 48 Hours (CVSS 7.0-8.9)\n"
                    "## P2 — Fix Within 1 Week (CVSS 4.0-6.9)\n"
                    "## P3 — Fix Within 1 Month (CVSS < 4.0)\n\n"
                    "## Security Hardening Recommendations\n"
                    "- Additional security measures not tied to specific vulns\n"
                    "- Security headers to add\n"
                    "- Monitoring and alerting to set up\n"
                    "- Dependency management policy\n"
                    "- Secret management improvements\n\n"
                    "Be SPECIFIC. Every recommendation must have a concrete action."
                ),
            ),
        ]),
    ],
    output_format=(
        "# Security Audit Report\n"
        "## Executive Summary\n"
        "## Attack Surface Map\n"
        "## Vulnerability Findings (by OWASP category)\n"
        "## CVSS Score Table\n"
        "## Remediation Roadmap\n"
        "## Compliance Mapping (OWASP/CWE/SANS)"
    ),
)


# =============================================================================
# GOAL: build-feature
# =============================================================================

_BUILD_FEATURE = GoalPipeline(
    name="build-feature",
    description="Full feature lifecycle — plan, implement, test, review",
    stages=[
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.PLANNER,
                label="Feature Planning",
                timeout=120,
                prompt_override=(
                    "You are a SOFTWARE ARCHITECT planning a feature implementation.\n\n"
                    "Analyze the feature request and produce:\n\n"
                    "## Requirements Analysis\n"
                    "- Core requirements (must-have)\n"
                    "- Edge cases to handle\n"
                    "- Out of scope (explicitly)\n\n"
                    "## Implementation Plan\n"
                    "1. Files to create or modify (with paths)\n"
                    "2. Data models / schema changes\n"
                    "3. API contracts (inputs, outputs, errors)\n"
                    "4. Dependencies needed\n\n"
                    "## Ordered Subtasks\n"
                    "Each subtask has:\n"
                    "- Clear description\n"
                    "- Files involved\n"
                    "- Done criteria\n"
                    "- Dependencies on other subtasks\n\n"
                    "## Test Strategy\n"
                    "- What to test (happy path, errors, edge cases)\n"
                    "- Test approach (unit, integration, e2e)\n\n"
                    "## Risks\n"
                    "- What could break\n"
                    "- Backward compatibility concerns\n\n"
                    "Read the existing code first to understand patterns and conventions."
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.CODER,
                label="Implementation",
                timeout=300,
                prompt_override=(
                    "You are implementing a feature based on the PLAN from Stage 1.\n\n"
                    "RULES:\n"
                    "1. Follow the plan exactly. If the plan is wrong, note it but still implement.\n"
                    "2. Follow existing code patterns, naming conventions, and style.\n"
                    "3. Handle errors and edge cases identified in the plan.\n"
                    "4. Write clean, production-ready code. No TODOs, no placeholders.\n"
                    "5. Use edit_file for existing files, write_file for new files.\n"
                    "6. After each file change, verify it with a quick read-back.\n\n"
                    "Output: list all files created/modified with a brief description of each change."
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.TESTER,
                label="Test Writing",
                timeout=0,
                prompt_override=(
                    "You are writing tests for the feature implemented in Stage 2.\n\n"
                    "COVERAGE REQUIREMENTS:\n"
                    "1. Happy path — the feature works as intended\n"
                    "2. Error cases — invalid inputs, missing data, permission errors\n"
                    "3. Edge cases — empty inputs, max values, concurrent access\n"
                    "4. Integration — the feature works with existing code\n\n"
                    "RULES:\n"
                    "- Use the project's existing test framework and patterns\n"
                    "- Tests must be deterministic (no random, no time-dependent)\n"
                    "- Prefer real objects over mocks where possible\n"
                    "- Each test should test ONE thing with a clear name\n"
                    "- Run the tests if possible (execute_shell) and fix failures\n\n"
                    "Output: test file paths and what each test covers."
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.REVIEWER,
                label="Final Review",
                timeout=120,
                prompt_override=(
                    "You are reviewing the feature implementation from Stages 2-3.\n\n"
                    "CHECK:\n"
                    "1. Does the implementation match the plan from Stage 1?\n"
                    "2. Code quality: clean, readable, follows conventions?\n"
                    "3. Error handling: all edge cases covered?\n"
                    "4. Tests: adequate coverage? All passing?\n"
                    "5. Any potential regressions?\n"
                    "6. Security: any new attack surface introduced?\n\n"
                    "Output:\n"
                    "- PASS or NEEDS WORK\n"
                    "- Summary of changes made\n"
                    "- Any remaining issues\n"
                    "- Suggested follow-up improvements (P3/nice-to-have)"
                ),
            ),
        ]),
    ],
    output_format=(
        "# Feature Implementation Report\n"
        "## Plan\n## Code Changes\n## Tests\n## Review Assessment"
    ),
)


# =============================================================================
# GOAL: debug
# =============================================================================

_DEBUG = GoalPipeline(
    name="debug",
    description="Systematic debugging — diagnose, fix, verify",
    stages=[
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.RESEARCHER,
                label="Investigation",
                timeout=0,
                prompt_override=(
                    "You are INVESTIGATING a reported bug.\n\n"
                    "1. Find the relevant code — use grep_search for error messages, function names\n"
                    "2. Read the code around the suspected area\n"
                    "3. Try to REPRODUCE: look for test files, run existing tests\n"
                    "4. Gather evidence:\n"
                    "   - Error messages, stack traces\n"
                    "   - Relevant code paths (trace from entry to error)\n"
                    "   - Recent changes (git_log, git_diff)\n"
                    "   - Similar patterns in the codebase\n\n"
                    "Output a DIAGNOSTIC REPORT:\n"
                    "- Symptoms observed\n"
                    "- Code path traced\n"
                    "- Evidence collected\n"
                    "- Initial hypotheses (ranked by likelihood)"
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.DEBUGGER,
                label="Root Cause Analysis",
                timeout=0,
                prompt_override=(
                    "You are performing ROOT CAUSE ANALYSIS on the investigation from Stage 1.\n\n"
                    "Apply the SCIENTIFIC METHOD:\n\n"
                    "1. Review the hypotheses from Stage 1\n"
                    "2. For each hypothesis:\n"
                    "   a. What would we expect to see if this is the cause?\n"
                    "   b. Examine the code to confirm or refute\n"
                    "   c. Verdict: confirmed, refuted, or inconclusive\n"
                    "3. Narrow down to the ACTUAL root cause\n"
                    "4. Explain the chain of events: trigger → root cause → symptom\n\n"
                    "Output:\n"
                    "- Root cause (1-2 sentences)\n"
                    "- Reasoning chain (how you got there)\n"
                    "- Exact location in code (file:line)\n"
                    "- Why the original code was wrong"
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.CODER,
                label="Fix Implementation",
                timeout=0,
                prompt_override=(
                    "You are FIXING the bug identified in Stage 2.\n\n"
                    "RULES:\n"
                    "1. Make the MINIMAL change needed to fix the root cause\n"
                    "2. Don't refactor surrounding code — fix only the bug\n"
                    "3. Handle the edge case that caused the bug\n"
                    "4. Check: does this fix introduce any new issues?\n"
                    "5. Use edit_file to make the change\n\n"
                    "Output:\n"
                    "- Exact change made (file:line, before/after)\n"
                    "- Why this fix is correct\n"
                    "- Any related areas that might need attention"
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.TESTER,
                label="Regression Tests",
                timeout=0,
                prompt_override=(
                    "You are writing REGRESSION TESTS for the bug fix from Stage 3.\n\n"
                    "Write tests that:\n"
                    "1. REPRODUCE the original bug (would fail before the fix)\n"
                    "2. VERIFY the fix works (passes after the fix)\n"
                    "3. Cover RELATED edge cases discovered during debugging\n"
                    "4. Prevent REGRESSION (future changes won't re-introduce this bug)\n\n"
                    "Run the tests (execute_shell) and verify they pass.\n\n"
                    "Output:\n"
                    "- Test file and test names\n"
                    "- What each test covers\n"
                    "- Test results (pass/fail)"
                ),
            ),
        ]),
    ],
    output_format=(
        "# Bug Fix Report\n"
        "## Symptoms\n## Root Cause\n## Fix Applied\n## Regression Tests"
    ),
)


# =============================================================================
# GOAL: explain
# =============================================================================

_EXPLAIN = GoalPipeline(
    name="explain",
    description="Architecture explanation — understand code without changing it",
    stages=[
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.RESEARCHER,
                label="Deep Research",
                timeout=0,
                prompt_override=(
                    "You are RESEARCHING code architecture for an explanation.\n\n"
                    "Explore THOROUGHLY — this is about understanding, not speed:\n\n"
                    "1. Map the file structure and module organization\n"
                    "2. Read key files: entry points, core modules, config\n"
                    "3. Trace data flow: input → processing → output\n"
                    "4. Map dependencies: imports, call graphs\n"
                    "5. Identify design patterns used\n"
                    "6. Note: what's the tech stack? what frameworks?\n\n"
                    "Read up to 15 files if needed — this is the explain goal, thoroughness matters.\n"
                    "Use grep_search to understand cross-module relationships.\n\n"
                    "Output: a detailed structured map of everything you found."
                ),
            ),
        ]),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.PLANNER,
                label="Architecture Explanation",
                timeout=120,
                prompt_override=(
                    "You are writing an ARCHITECTURE EXPLANATION based on Stage 1 research.\n\n"
                    "Create a clear, educational explanation:\n\n"
                    "## Overview\n"
                    "What does this codebase do? 2-3 sentences.\n\n"
                    "## Architecture Diagram\n"
                    "ASCII or Mermaid diagram showing components and relationships.\n\n"
                    "## Key Components\n"
                    "For each major module/class:\n"
                    "- What it does (purpose)\n"
                    "- How it connects to other components\n"
                    "- Key functions/methods\n\n"
                    "## Data Flow\n"
                    "Step-by-step: how data moves through the system.\n\n"
                    "## Design Decisions\n"
                    "WHY things are designed this way. What tradeoffs were made.\n\n"
                    "## Extension Points\n"
                    "Where and how you'd extend this system.\n\n"
                    "DO NOT suggest code changes. This is explanation only.\n"
                    "Use analogies where helpful. Write for a developer joining the team."
                ),
            ),
        ]),
    ],
    output_format=(
        "# Architecture Explanation\n"
        "## Overview\n## Diagram\n## Components\n## Data Flow\n## Design Decisions"
    ),
)


# =============================================================================
# GOAL: optimize
# =============================================================================

_OPTIMIZE = GoalPipeline(
    name="optimize",
    description="Performance optimization — profile, refactor, verify",
    stages=[
        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.RESEARCHER,
                label="Performance Profiling",
                timeout=0,
                prompt_override=(
                    "You are PROFILING code for performance optimization.\n\n"
                    "Analyze through STATIC ANALYSIS (read the code, trace execution):\n\n"
                    "## Algorithmic Complexity\n"
                    "- Identify Big-O of key operations\n"
                    "- Flag O(n^2) or worse where O(n) or O(n log n) is possible\n"
                    "- Nested loops over large datasets\n\n"
                    "## Redundant Computation\n"
                    "- Values computed multiple times (cache opportunity)\n"
                    "- Unnecessary iterations, repeated lookups\n"
                    "- Work done that's never used\n\n"
                    "## I/O Bottlenecks\n"
                    "- Synchronous I/O in hot paths\n"
                    "- N+1 queries (database, API calls)\n"
                    "- Missing connection pooling\n"
                    "- Large file reads into memory\n\n"
                    "## Memory\n"
                    "- Large allocations, unbounded growth\n"
                    "- Unnecessary copies vs references\n"
                    "- Missing generators/iterators for large sequences\n\n"
                    "Output: ranked list of optimization opportunities with estimated impact."
                ),
            ),
        ]),

        # Stage 2: Parallel refactoring
        GoalStage(
            steps=[
                GoalStep(
                    worker_type=WorkerType.REFACTORER,
                    label="Algorithm & Logic Optimization",
                    timeout=0,
                    prompt_override=(
                        "You are optimizing ALGORITHMS AND LOGIC based on Stage 1 profiling.\n\n"
                        "Focus on:\n"
                        "1. Reducing time complexity (better algorithms, data structures)\n"
                        "2. Adding memoization/caching for repeated computations\n"
                        "3. Batch operations instead of one-at-a-time\n"
                        "4. Lazy evaluation where appropriate\n"
                        "5. Early exits, short-circuit evaluation\n\n"
                        "For each optimization:\n"
                        "- BEFORE: the original code + its Big-O\n"
                        "- AFTER: the optimized code + its Big-O\n"
                        "- IMPACT: estimated speedup\n\n"
                        "Use edit_file to apply changes. Do NOT change behavior, only performance."
                    ),
                ),
                GoalStep(
                    worker_type=WorkerType.REFACTORER,
                    label="I/O & Resource Optimization",
                    timeout=0,
                    prompt_override=(
                        "You are optimizing I/O AND RESOURCE USAGE based on Stage 1 profiling.\n\n"
                        "Focus on:\n"
                        "1. Async where beneficial (I/O-bound operations)\n"
                        "2. Connection pooling, request batching\n"
                        "3. Streaming vs buffering for large data\n"
                        "4. Memory efficiency: generators, iterators, __slots__\n"
                        "5. Reducing unnecessary file/network operations\n\n"
                        "For each optimization:\n"
                        "- BEFORE: the original code + resource usage\n"
                        "- AFTER: the optimized code + resource usage\n"
                        "- IMPACT: estimated improvement\n\n"
                        "Use edit_file to apply changes. Do NOT change behavior, only performance."
                    ),
                ),
            ],
            merge_strategy="concatenate",
        ),

        GoalStage(steps=[
            GoalStep(
                worker_type=WorkerType.TESTER,
                label="Performance Verification",
                timeout=0,
                prompt_override=(
                    "You are VERIFYING that optimizations from Stage 2 didn't break anything.\n\n"
                    "1. Run existing tests (execute_shell) — ALL must pass\n"
                    "2. Write performance benchmarks where possible:\n"
                    "   - Time the optimized code paths\n"
                    "   - Compare with expected complexity\n"
                    "3. Write regression tests for any refactored code\n"
                    "4. Verify behavior is IDENTICAL to pre-optimization\n\n"
                    "Output:\n"
                    "- Existing test results (pass/fail)\n"
                    "- New benchmark results\n"
                    "- Any issues found\n"
                    "- Optimization summary with measured improvements"
                ),
            ),
        ]),
    ],
    output_format=(
        "# Optimization Report\n"
        "## Profile Summary\n## Optimizations Applied\n## Benchmarks\n## Verification"
    ),
)


# =============================================================================
# GOAL REGISTRY
# =============================================================================

GOAL_REGISTRY: dict[str, GoalPipeline] = {
    "code-review": _CODE_REVIEW,
    "cyber-security": _CYBER_SECURITY,
    "build-feature": _BUILD_FEATURE,
    "debug": _DEBUG,
    "explain": _EXPLAIN,
    "optimize": _OPTIMIZE,
}


def get_goal(name: str) -> GoalPipeline | None:
    """Get a goal pipeline by name. Returns None for 'general' (default flow)."""
    if name == "general":
        return None
    return GOAL_REGISTRY.get(name)


def list_goals() -> list[dict[str, str]]:
    """List all available goals with descriptions."""
    goals = [{"name": "general", "description": "Default — queen routes everything automatically"}]
    for name, pipeline in GOAL_REGISTRY.items():
        goals.append({"name": name, "description": pipeline.description})
    return goals

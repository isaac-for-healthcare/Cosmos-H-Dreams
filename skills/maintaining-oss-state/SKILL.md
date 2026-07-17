---
name: maintaining-oss-state
description: Maintain Cosmos-H-Dreams's OSS-release state — the LICENSE / NOTICE / THIRD-PARTY-NOTICES / REUSE.toml / LICENSES/ / CONTRIBUTING.md collateral that satisfies the Cosmos-H-Dreams OSRB review, the per-file SPDX headers, the third-party dependency manifest in THIRD-PARTY-NOTICES, and the pyproject.toml + uv.lock dependency pins. Use when adding or upgrading a runtime dependency, vendoring third-party source into the repo, adding a new first-party source file (any .py / .pyx / .pyi / .c / .cc / .cpp / .h / .hpp / .cu / .cuh / .sh / .proto / Dockerfile), reviewing whether a change requires reopening the OSRB bug or filing a self-cert, or triaging a reuse-lint CI failure.
---

# Maintaining Cosmos-H-Dreams's OSS state

Cosmos-H-Dreams is released to the public under Apache-2.0 (OSRB Bug
[6460079](https://nvbugspro.nvidia.com/bug/6460079)). The repo carries a
fixed set of collateral that the OSRB approved on, and a `reuse-lint` CI
workflow that fails the build if that collateral drifts. This skill is
the map for keeping the collateral consistent — what each file is for,
what edits trigger which downstream paperwork, and which CI gates catch
what.

> OSRB Bug [6460079](https://nvbugspro.nvidia.com/bug/6460079) is the
> canonical record for every dependency-approval operation described
> below. Reopen it and amend its dependency list whenever a change in
> this skill says to.

## TL;DR

- **Six files at repo root + one CI workflow define OSS state:**
  `LICENSE`, `LICENSES/`, `NOTICE`, `THIRD-PARTY-NOTICES`,
  `REUSE.toml`, `CONTRIBUTING.md`, and
  `.github/workflows/reuse-lint.yml`. Touch any of them with the same
  care you'd give to a public API.
- **`LICENSE` is Apache-2.0 with a short licensing-posture preamble**
  explaining that the entire repo — including the vendored
  `flashdreams/` subtree — is Apache-2.0, then reproduces the full
  Apache-2.0 text. CI verifies the canonical Apache-2.0 sentinel
  strings are present; the preamble prose is not lint-checked (this
  file changes rarely — see the change-log review path instead).
- **`NOTICE` is the minimal Apache 2.0 §4(d) notice** — NVIDIA
  copyright + pointers to `LICENSE`, `LICENSES/`, and
  `THIRD-PARTY-NOTICES`, plus a one-line acknowledgement of the
  vendored `flashdreams/` subtree. It is *not* the full attribution
  document.
- **`THIRD-PARTY-NOTICES` is the full per-dependency attribution
  document** — direct runtime deps (split per workspace package),
  reference architectures, and source-level redistributions, each with
  SPDX identifier and upstream URL.
- **Every first-party source file carries an inline SPDX header.**
  `REUSE.toml`'s `**` aggregate keeps the lint green for files that
  can't carry one (config, assets, binaries), but `reuse-lint`'s
  "Inline SPDX headers on first-party source files" step rejects any
  new `.py` / `.c` / `.cpp` / `.cu` / `.sh` / `.proto` / `Dockerfile`
  / etc. without the inline tag.
- **Direct deps are mirrored in three places:** the workspace member's
  `pyproject.toml` `dependencies`, the resolved `uv.lock` pin, and the
  `THIRD-PARTY-NOTICES` "Direct runtime dependencies" table. All
  three must agree.
- **The one third-party subtree physically present in the repo is
  `flashdreams/`** (vendored from NVIDIA FlashDreams, Apache-2.0). It is
  documented in `THIRD-PARTY-NOTICES` "Source-level redistributions" and
  in `NOTICE`. Because it carries the *same* Apache-2.0 license as the
  rest of the repo, it needs no `REUSE.toml` `override` and no extra
  `LICENSES/` text — its files keep the standard Apache-2.0 SPDX header.
- **OSRB Bug 6460079 is the source of truth for which deps are
  "covered" by the approval.** Adding a new direct dep = reopen the bug
  and amend the dependency list. Adding a transitive that only matters
  because the SBOM scanner flagged it = either reopen + amend, or file a
  self-cert. Dev-only transitives → SBOM correction (not shipped).

## 1. The OSS-collateral file set

| File / path | Role | OSRB anchor |
|---|---|---|
| `LICENSE` | Apache-2.0 with a short licensing-posture preamble (the whole repo, incl. vendored `flashdreams/`, is Apache-2.0) followed by the canonical Apache-2.0 v2.0 text. CI verifies the canonical Apache-2.0 sentinel strings are present (preamble prose is not lint-gated). | OSRB license review |
| `LICENSES/Apache-2.0.txt` | REUSE 3.3 license-bundle copy of the canonical Apache-2.0 text (no preamble — must remain reusable verbatim by REUSE tooling). | OSRB license review |
| `LICENSES/BSD-3-Clause.txt` | Full BSD-3-Clause text, bundled because several runtime dependencies (torch, numpy, nvidia-ml-py, aiortc) are BSD-3-Clause and are attributed in `THIRD-PARTY-NOTICES`. | OSRB attribution review |
| `NOTICE` | Apache 2.0 §4(d) minimal notice — NVIDIA copyright + pointers to `LICENSE`, `LICENSES/`, and `THIRD-PARTY-NOTICES`, plus the one-line vendored-`flashdreams/` acknowledgement. Carried forward verbatim by downstream redistributions. | Apache-2.0 §4(d) |
| `THIRD-PARTY-NOTICES` | Full per-dependency attribution: per-package Direct runtime deps + Reference architectures + Source-level redistributions. The source of truth for the third-party manifest. | OSRB review (canonical attribution doc) |
| `REUSE.toml` | REUSE 3.3 aggregate / override annotations for files without inline SPDX. | OSRB REUSE-compliance review |
| `CONTRIBUTING.md` | Apache-2.0-only contribution statement + DCO v1.1 reproduction + "Signing Your Work" subsection (with `git commit -s` and `--signoff`) + IP-review reference. | OSRB DCO template |
| `.github/workflows/reuse-lint.yml` | CI gate enforcing REUSE 3.3 compliance, presence of the five core OSRB collateral files (`LICENSE`, `LICENSES/Apache-2.0.txt`, `CONTRIBUTING.md`, `NOTICE`, `REUSE.toml`), canonical Apache-2.0 text in `LICENSE` and `LICENSES/Apache-2.0.txt`, `CONTRIBUTING.md` DCO/sign-off reference, inline SPDX headers (incl. `.sh` / `.proto` / Dockerfile), and no legacy proprietary banners. Content-shape of the `LICENSE` preamble / `CONTRIBUTING.md` policy text / `THIRD-PARTY-NOTICES` sections is reviewed manually, not lint-checked. | (enforces the above) |

The CI workflow runs on every PR, every push to `main`, and inside the
GitHub merge queue. A failed `reuse-lint` blocks merge — never bypass it,
fix the underlying file.

## 2. Per-file SPDX headers

Every first-party source file starts with the inline SPDX header. The
exact wording is enforced by `reuse-lint`'s "Inline SPDX headers on
first-party source files" step (looks for `SPDX-License-Identifier` in
the first 20 lines).

**Python / shell / TOML / YAML** (`#` line comments):

```python
# SPDX-FileCopyrightText: Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**C / C++ / CUDA** (`//` line comments): same two SPDX tags + the same
Apache-2.0 preamble, with `//` swapped for `#`.

Rules:

- `<YEAR>` is the **current calendar year** for newly created files
  (use the system clock — *not* a model-training-cutoff year). For
  files being **edited**, leave the year alone — it reflects original
  authorship, not last-touched.
- The two SPDX tags (`SPDX-FileCopyrightText` + `SPDX-License-Identifier`)
  are the load-bearing part. The Apache-2.0 preamble is house style;
  the CI gate only checks for `SPDX-License-Identifier` in the first 20
  lines, but the long form is what every existing file carries, so
  match it.
- External contributors add their **own** copyright line *above* the
  NVIDIA line — keep both. See `CONTRIBUTING.md`.
- The vendored `flashdreams/` subtree is NVIDIA-authored Apache-2.0 and
  its files carry the standard NVIDIA Apache-2.0 header — no special
  dual-copyright handling. If you ever redistribute *modified* upstream
  Apache-2.0 source from a third party, add that party's
  `SPDX-FileCopyrightText` line *above* the NVIDIA one and keep both.

### What's exempt (handled by `REUSE.toml`)

- Binary assets (`assets/**.png`, `.jpg`, `.jpeg`, `.webp`, `.mp4`,
  `.gif`, `.svg`) and lock files (`uv.lock`).
- Documentation (`**.md`, `**.rst`, `docs/**`).
- Configuration / build / CI / shell-wrapper files enumerated in the
  `REUSE.toml` config block.

If a tracked source file genuinely cannot carry an inline header (a
tooling-generated artifact, an asset, a config file), extend `REUSE.toml`
rather than fighting the lint.

## 3. `REUSE.toml` — aggregate vs override

`REUSE.toml` is the REUSE 3.3 manifest that fills gaps the inline SPDX
header convention can't.

Precedence rules (from `REUSE.toml`):

- **`precedence = "aggregate"`** — declared license merges with any
  inline SPDX header the file carries. Used for the project-wide default
  (`path = "**"` → Apache-2.0), for documentation / config / build /
  asset blocks, and for the `flashdreams/**` subtree (same Apache-2.0
  license as the rest of the repo).
- **`precedence = "override"`** — declared license *replaces* any inline
  SPDX header. Cosmos-H-Dreams does not currently use any `override`
  block because it vendors no third-party source under a *different*
  license. Reach for `override` only if you vendor upstream source whose
  own (non-Apache-2.0) banners must stay authoritative — see §6.

When to add an annotation block:

| Scenario | Block style | precedence |
|---|---|---|
| Add a new first-party source-file type covered by the default `**` rule | (nothing — default covers it) | n/a |
| Add a new asset / config / generated-output path that can't carry an inline header | new `[[annotations]]` block, copyright + Apache-2.0 | `aggregate` |
| Add a new third-party source subtree under a *different* license (banners we want to keep) | new `[[annotations]]` block, copyright = upstream, SPDX = upstream's license | `override` |
| Add a redistributed-and-modified upstream Apache-2.0 file | new block listing dual `SPDX-FileCopyrightText` (NVIDIA + upstream) | `aggregate` |

More specific paths win — order the annotations so specific paths come
*after* general ones.

## 4. `NOTICE` vs `THIRD-PARTY-NOTICES` — what goes where

Two distinct files. Mixing them up is the most common OSS-state mistake.

### `NOTICE` — minimal, downstream-propagated

`NOTICE` exists to satisfy **Apache 2.0 §4(d)**: any derivative work must
carry a readable copy of the upstream `NOTICE` text. Keep it small so
downstream consumers do not pay an unreasonable carry-forward cost.
Shape:

```
NVIDIA Cosmos-H-Dreams
Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This product is licensed under the Apache License, Version 2.0; the
full license text is reproduced in LICENSE at the repository root and
in LICENSES/Apache-2.0.txt.

One subtree physically vendored into this repository (flashdreams/,
sourced from NVIDIA FlashDreams) carries the same Apache-2.0 license.

Third-party software attributions, source-level redistribution
disclosures, and the full per-dependency license inventory are
documented in THIRD-PARTY-NOTICES at the repository root.
```

Do not enumerate transitive dependencies, SPDX tables, or per-package
attributions in `NOTICE`. Those go in `THIRD-PARTY-NOTICES`.

### `THIRD-PARTY-NOTICES` — full per-dependency manifest

`THIRD-PARTY-NOTICES` is the consumer-facing attribution document and
the source of truth for the third-party manifest. It has named
sections; do not invent new ones without a corresponding `REUSE.toml`
change where relevant.

```
NVIDIA Cosmos-H-Dreams — Third-Party Notices
Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

  Preamble explaining the dynamic-import / no-source-redistribution
  default and pointing readers at the Source-level redistributions
  section at the bottom.

================================================================================
Direct runtime dependencies — flashdreams package
================================================================================

  <name>  <SPDX>  <upstream-URL>
  ... one row per direct dep of the flashdreams core package ...

================================================================================
Direct runtime dependencies — flash-cosmosHDreams package
================================================================================

  <name>  <SPDX>  <upstream-URL>
  ... one row per direct dep of the cosmosHDreams serving package ...

================================================================================
Reference architectures
================================================================================

  Wan 2.1 / Wan 2.2   Apache-2.0   https://github.com/Wan-Video/Wan2.1
      <one paragraph explaining the reference-architecture relationship
       and confirming weights/sources aren't redistributed>

================================================================================
Source-level redistributions
================================================================================

  <one block per subtree of third-party source physically in the repo —
   currently just flashdreams/ (Apache-2.0, from NVIDIA FlashDreams)>
```

Rules:

- **Column 1 = exact PyPI / upstream name.** Match the workspace
  member's `pyproject.toml` `dependencies =` spelling
  (e.g., `opencv-python-headless`, not `opencv`).
- **Column 2 = SPDX identifier** (from <https://spdx.org/licenses/>). For
  dual-licensed packages use comma-separated SPDX IDs in alphabetical
  order, e.g., `MIT, MPL-2.0` for tqdm.
- **Column 3 = upstream source URL**, not the PyPI page.
- **Only direct deps go in the top tables.** Split them by workspace
  package (`flashdreams` core vs `flash-cosmosHDreams` serving) so the
  manifest maps 1:1 onto the two `pyproject.toml` files. Transitives
  stay out unless they're material enough to flag.
- **Reference architectures get their own block** with a 2–4 line
  explanation (we implement the architecture; we don't redistribute the
  upstream code or weights).
- **The "Source-level redistributions" section is the only place
  physically-present third-party source is acknowledged.** Each block:
  Path (from repo root), License (SPDX + pointer to
  `LICENSES/<SPDX>.txt` when the license differs from Apache-2.0),
  Upstream URL, and a paragraph explaining what was modified vs. what's
  upstream code. Cosmos-H-Dreams has exactly one such block today:
  `flashdreams/`.

## 5. Adding or upgrading a runtime dependency

This is the highest-frequency OSS-state edit. It touches **four** places:

1. The workspace member's `pyproject.toml` that needs the dep
   (`flashdreams/pyproject.toml` for the core package,
   `cosmosHDreams/pyproject.toml` for the serving package) — add to
   `dependencies = [...]`. Pin a floor (`>=`) on semver-stable packages;
   pin tightly (`==`) only when the upstream API is known-unstable across
   minor versions.
2. `uv.lock` — regenerate with `uv lock` so the hash-pinned resolved
   version lands in the lockfile.
3. `THIRD-PARTY-NOTICES` — add a row to the matching package's "Direct
   runtime dependencies" table with `name  SPDX  upstream-URL`.
4. The OSRB contribution bug — **reopen the bug and amend the approved
   dependency list** with the new (name, version, license, URL) row.
   Per OSRB policy, for ongoing contributions the previously-approved
   contribution bug must be reopened whenever a new package is added to
   the product delivery, a previously-approved package changes its
   license, or the use of a previously-approved component changes.

### Pre-add checklist

- [ ] **Confirm the SPDX license**. Read the upstream LICENSE file
      (don't trust GitHub's auto-detected badge). MPL-2.0, LGPL,
      AGPL, GPL, EPL, MS-PL all carry copyleft conditions; loop in
      legal *before* adding.
- [ ] **Confirm the dep is published from PyPI**, not from a private
      index. If it isn't on PyPI, write `[tool.uv.sources]` with care
      and flag for OSRB review.
- [ ] **Check for security advisories** (Snyk, BDSA, NVD). The repo
      has carried floor pins for security reasons before
      (`urllib3>=2.7.0` for the botocore/requests CVE chain). If a
      floor is needed, leave a one-line comment in `pyproject.toml`
      explaining why.
- [ ] **MPL-2.0 / weak-copyleft**: confirm dynamic import only, no
      modifications, no source redistribution. If any of those don't
      hold, the dep needs to be vendored (see §6) and OSRB review is
      mandatory.
- [ ] **Codec / crypto**: if the new dep implements an audio/video
      codec or encryption (as PyNvVideoCodec / aiortc do), confirm the
      OSRB export-control questions are still answered correctly.

### Upgrading an existing dep version

- **Same license, same SPDX, version-bump only** → update
  `pyproject.toml` floor (if needed), regen `uv.lock`.
  `THIRD-PARTY-NOTICES` may not need a touch (we don't pin exact
  versions there). OSRB bug does **not** need to be reopened (policy
  explicit: "version updates without licensing changes don't require
  reopening").
- **License change between versions** → treat as a new dep:
  reopen the OSRB bug, update `THIRD-PARTY-NOTICES`, possibly re-check
  codec/crypto questions.
- **Major version bump that changes the dep's *use*** (e.g.,
  switching from sync-only to async-only, or adding a new transitive
  family) — reopen the OSRB bug even if the SPDX hasn't changed
  ("The use of a previously approved component changed").

### `uv.lock` hygiene

- Always regenerate the lock from the workspace root: `uv lock`.
- Commit `pyproject.toml` and `uv.lock` together — they are
  jointly maintained.
- If a dep's transitives shift in a way that drops or adds a
  *direct-of-direct* (e.g., `httpx` → drops `certifi`), the new
  closure is what the OSRB SBOM scanner will see — re-run the
  scanner after the merge so anything new gets caught.

## 6. Adding a third-party source-level redistribution

Vendoring upstream source under a *different* license into the repo is
heavier than a runtime dep — it touches **six** places. (Cosmos-H-Dreams
does not do this today; `flashdreams/` is Apache-2.0 and needs none of
the license-divergence machinery below. Follow these steps only if you
introduce non-Apache upstream source.)

1. **Physically place the source** under a subtree path that signals
   it's third-party. Keep the upstream banners verbatim in the file
   headers — don't replace them with NVIDIA SPDX headers.
2. **Add the full license text** under `LICENSES/<SPDX>.txt`.
3. **Extend `REUSE.toml`** with an `override` annotation for the
   subtree (license = upstream SPDX, copyright = upstream copyright).
4. **Add a "Source-level redistributions" block in
   `THIRD-PARTY-NOTICES`** — path, license + pointer to
   `LICENSES/<SPDX>.txt`, upstream URL, one paragraph describing what
   we modified vs. upstream. **Also update `NOTICE`** to add a
   one-line entry, since physically-redistributed third-party source is
   one of the things downstream consumers must see when carrying our
   Apache 2.0 §4(d) notice forward. **Also update the `LICENSE`
   preamble** to cross-reference the new `LICENSES/<SPDX>.txt`.
5. **OSRB bug** — reopen + add the dependency row *and* note the
   source-level redistribution in a comment. Filing an OSRB bug for the
   upstream project itself may be required if it has its own OSRB
   process.
6. **`reuse-lint` exclusion** in `.github/workflows/reuse-lint.yml`'s
   "Inline SPDX headers on first-party source files" step — extend
   the `excludes` regex to skip the new subtree, since upstream
   banners use the upstream license, not Apache-2.0.

## 7. Adding a new first-party source file

Easy path — `reuse-lint` will fail the PR if you skip a step.

1. Open the file with the SPDX header (see §2). The current calendar
   year for newly authored files.
2. Save / `git add`. No `REUSE.toml` change needed — the `**`
   default rule covers it.
3. If you have introduced an `override` subtree (see §6) and the file
   lives under it, it inherits the upstream license — only do this when
   the file genuinely *is* upstream-derived, not because it's convenient.

The `reuse-lint` "No NVIDIA proprietary banners" step will reject any
file that still carries the legacy NVIDIA proprietary banner. If you
ported source from an internal repo, strip the old banner and replace
with the Apache-2.0 SPDX header.

## 8. OSRB bug interaction matrix

| Change | OSRB action |
|---|---|
| Bump version, same license | None (policy explicit). |
| Bump version, license changed | Reopen bug, amend dependency list. |
| Add new direct dep | Reopen bug, amend dependency list, update `THIRD-PARTY-NOTICES`. |
| Add new transitive flagged by SBOM scanner | Reopen bug (preferred), OR file a self-cert. |
| Add dev/test-only transitive (`[dev]` extra) flagged by scanner | File **SBOM correction** — not in product delivery. Self-cert as fallback. |
| Vendor third-party source physically into repo | Reopen bug + comment thread; possibly file a sub-OSRB bug for the upstream project. |
| Remove a dep | Update `pyproject.toml`, `uv.lock`, `THIRD-PARTY-NOTICES`. No OSRB action — removal doesn't add new attack surface. |
| Drop a previously-approved transitive (closure shift) | Update `THIRD-PARTY-NOTICES` if it was listed; no OSRB action required. |

### MPL-2.0 in particular

NVIDIA accepts MPL-2.0 use when **all three** hold:

1. **No modifications** to the upstream MPL-2.0 source.
2. **No source redistribution** — the dep is consumed from PyPI at
   install time, not vendored.
3. **Dynamic linking only** (Python `import`).

Document this trio explicitly on every MPL-2.0 self-cert ticket. If any
of the three fails, the dep needs a regular Use bug.

## 9. Updating CONTRIBUTING.md

The DCO v1.1 text is reproduced verbatim in `CONTRIBUTING.md`.
**Do not paraphrase, summarize, or "modernize" it** — the `reuse-lint`
collateral step looks for the exact pattern
`Developer.{1,40}Certificate.{1,10}of.{1,10}Origin|Signed-off-by|sign-off`,
and OSRB approval is on the *verbatim* text.

When extending CONTRIBUTING.md:

- Keep the DCO section anchored at `## Developer Certificate of Origin
  (DCO)` — the README and external docs link to it by anchor.
- The SPDX header preamble in `CONTRIBUTING.md` doubles as the
  agent-and-human source for what every new source file's header should
  look like. Update it and `python-docstring-style/SKILL.md` together.
- The IP-review-process reference in `CONTRIBUTING.md` is an OSRB
  pointer — don't change it without OSRB sign-off.

## 10. CI gates — what `reuse-lint` enforces

The workflow has two jobs. Read `.github/workflows/reuse-lint.yml` if
you need to add a new gate.

The gate set is intentionally narrow — these files change rarely and
the cost of over-fitted CI (false positives, sweeping rewrites
needed when wording shifts) exceeds the benefit. Treat the lint as a
backstop for structural regressions (missing collateral file, missing
SPDX header, legacy banner) and trust human review for content shape
(preamble references, contribution policy wording, attribution-table
sections).

**`reuse` job** (`fsfe/reuse-action@v5`):

- REUSE 3.3 lint: every tracked file has an SPDX identifier, either
  inline or via `REUSE.toml`. New files without coverage fail.

**`collateral` job** (custom bash):

1. `LICENSE` and `LICENSES/Apache-2.0.txt` both contain the canonical
   Apache-2.0 sentinel strings. (`LICENSE` may carry a licensing-posture
   preamble in front of the Apache-2.0 body; `LICENSES/Apache-2.0.txt`
   must stay preamble-free so REUSE tooling can reuse it verbatim.)
2. The five core OSRB collateral files exist at the repo root:
   `LICENSE`, `LICENSES/Apache-2.0.txt`, `CONTRIBUTING.md`, `NOTICE`,
   `REUSE.toml`. (`LICENSES/BSD-3-Clause.txt` and `THIRD-PARTY-NOTICES`
   also need to be present per OSRB approval, but are not lint-gated —
   they change slowly and a manual review catches drift sooner than the
   cost of over-fitted CI would justify.)
3. `CONTRIBUTING.md` references the DCO / sign-off.
4. Every tracked source file (`.py`, `.pyx`, `.pyi`, `.c`, `.cc`,
   `.cpp`, `.cxx`, `.h`, `.hh`, `.hpp`, `.hxx`, `.cu`, `.cuh`,
   `.inl`, `.sh`, `.proto`, `Dockerfile` / `*.dockerfile`) carries
   an inline `SPDX-License-Identifier` in its first 20 lines — with
   any documented `override`-subtree exclusions.
5. No file contains the legacy NVIDIA proprietary-banner sentinel
   phrases checked by `.github/workflows/reuse-lint.yml`.

Triggers: every PR, every push to `main`, and every merge-queue group.

> Housekeeping: the `excludes` regex in the inline-SPDX step and the
> path skipped by the "No NVIDIA proprietary banners" step should name
> paths that actually exist in this repo. Prune any leftover upstream
> paths so the exclusions document real exemptions, not phantom ones.

## 11. Common pitfalls

- **Editing `LICENSE` without preserving the canonical Apache-2.0
  body** — the collateral step greps for sentinel strings in both
  `LICENSE` and `LICENSES/Apache-2.0.txt`. Keep the Apache-2.0 text
  intact in both; the preamble lives only in `LICENSE`.
- **Adding a new direct dep and forgetting `THIRD-PARTY-NOTICES`**.
  The lint won't catch this (the file is free-form prose). Add the
  attribution row in the same commit that touches `pyproject.toml` /
  `uv.lock` and the matching PR.
- **Touching `NOTICE` when only `THIRD-PARTY-NOTICES` should
  change**. `NOTICE` is the small Apache 2.0 §4(d) file that
  downstream consumers carry forward verbatim — keep it minimal.
  Routine dep additions belong in `THIRD-PARTY-NOTICES`. Touch
  `NOTICE` only when (a) the year on line 2 rolls forward, (b) a new
  source-level redistribution subtree appears (rare), or (c) the
  pointer text needs to mention a new top-level OSS file.
- **Using single-quoted SPDX-FileCopyrightText**. REUSE is forgiving;
  the rest of the repo uses double-quoted strings. Mismatch breaks
  human grep, not the lint.
- **Year stamp drift**. The SPDX header year is *when the file was
  first authored*, not when it was last touched — never mass-rewrite
  the year across the tree. The exceptions are line 2 of `NOTICE`,
  line 2 of `THIRD-PARTY-NOTICES`, and the project copyright in the
  `LICENSE` preamble, which are the project's overall copyright year
  and are allowed to roll forward annually.
- **Keeping a `LICENSES/<SPDX>.txt` for a license no longer used
  anywhere.** REUSE flags unused license texts. If you drop the last
  dependency (or subtree) under a given license, remove the now-orphan
  license file too.
- **Skipping `uv lock` after a `pyproject.toml` edit.** The lockfile
  is the source of truth for what consumers actually install; an
  out-of-sync `uv.lock` is a real bug, not a cosmetic one.
- **Adding a dep "just for tests" without `[dev]` extra placement.**
  If a dep lives in the `dev` extra, it's not in the product delivery
  and is out of OSRB scope per the policy bullet. If it's in
  `dependencies = [...]`, it ships to every consumer — OSRB-scoped.
  The `[dev]`/optional extras in `flashdreams/pyproject.toml` and
  `cosmosHDreams/pyproject.toml` are the seams.

## 12. Scaffolding checklist — full operations

**Add a new direct runtime dep `foo` (semver-stable, Apache/MIT/BSD):**

1. Add `"foo>=X.Y"` to the right workspace's `pyproject.toml`
   `dependencies = [...]`.
2. `uv lock` from the workspace root; commit `pyproject.toml` +
   `uv.lock` together.
3. Add a row to the matching package's `THIRD-PARTY-NOTICES` "Direct
   runtime dependencies" table: `foo  <SPDX>  <upstream-URL>`.
4. Reopen the OSRB contribution bug, amend the approved dependency
   list with the new row, and ping for re-approval.
5. PR + CI passes `reuse-lint` → merge.

**Add a new direct runtime dep `bar` (MPL-2.0 / LGPL / other
weak-copyleft):**

1. Verify dynamic-import / no-mod / no-redistribute trio (§8). If any
   fails, stop and engage OSRB.
2. Same four steps above.
3. *Also* file a self-cert ticket for the weak-copyleft dep per OSRB
   policy.

**Bump dep `baz` from 2.x to 3.x (same license):**

1. Verify SPDX unchanged at the new version.
2. Update `pyproject.toml` floor if API contract requires.
3. `uv lock`; commit `pyproject.toml` + `uv.lock`.
4. No OSRB action.
5. (Optional) Update `THIRD-PARTY-NOTICES` if its row drifts
   (e.g., URL changed).

**Vendor upstream `qux` (BSD-3) into a third-party subtree:**

1. Drop source in with upstream banners preserved.
2. Add `LICENSES/BSD-3-Clause.txt` if not already present.
3. New `[[annotations]]` block in `REUSE.toml` with
   `precedence = "override"`, BSD-3 SPDX, upstream copyright, path
   `"<subtree>/**"`.
4. New block in `THIRD-PARTY-NOTICES` "Source-level
   redistributions" — path,
   `License: BSD-3-Clause (see LICENSES/BSD-3-Clause.txt)`,
   upstream URL, modification paragraph. Also add a line to `NOTICE`
   and a pointer in the `LICENSE` preamble.
5. Extend the `excludes` regex in
   `.github/workflows/reuse-lint.yml`'s inline-SPDX step.
6. Reopen the OSRB contribution bug; file a sub-OSRB if `qux` has its
   own project-level OSRB process.

**Triage a `reuse-lint` failure:**

1. Read the failed step name.
2. `REUSE 3.3 compliance` → run `pipx run reuse lint` locally; add
   inline SPDX or extend `REUSE.toml`.
3. `LICENSE carries canonical Apache-2.0 text` → confirm the sentinel
   strings are intact in `LICENSE` and `LICENSES/Apache-2.0.txt`.
4. `Required OSRB collateral present` → recreate the missing file
   from history (`git log -- <file>` to find the original commit).
5. `CONTRIBUTING.md references the DCO` → restore the DCO section
   anchor and verbatim text.
6. `Inline SPDX headers on first-party source files` → the step
   prints every offending path as a GitHub annotation. Add the header
   to each.
7. `No NVIDIA proprietary banners` → strip the legacy banner from
   the listed file(s), replace with the Apache-2.0 SPDX header.

## 13. Where this maps in the codebase

| Question | File / pointer |
|---|---|
| What's the canonical Apache-2.0 text? | `LICENSE` / `LICENSES/Apache-2.0.txt` |
| What deps does Cosmos-H-Dreams ship? | `THIRD-PARTY-NOTICES` "Direct runtime dependencies" tables + `flashdreams/pyproject.toml` + `cosmosHDreams/pyproject.toml` |
| What does a SPDX header look like? | `CONTRIBUTING.md` (SPDX header section) |
| Where do I declare a config / asset file's license? | `REUSE.toml` |
| Where do I record vendored upstream source? | `THIRD-PARTY-NOTICES` "Source-level redistributions" + `NOTICE` (+ `REUSE.toml` `override` block only if the license differs from Apache-2.0) |
| What does the CI gate enforce? | `.github/workflows/reuse-lint.yml` (read top to bottom) |
| Which deps did OSRB approve? | The dependency list on OSRB Bug 6460079 (kept in sync with `THIRD-PARTY-NOTICES`) |

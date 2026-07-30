# meetscribe — open-source release prep (2026-07-30)

**Goal.** Take `LucianoLupo/meetscribe` from private-and-welded-to-one-Mac to a public
repo a stranger can clone, build, sign, and run.

**Starting state.** master = `086c943`, in sync with origin. v1 complete (Phases 0–5),
launchd daemon live, 36/36 tests + clippy clean. Streaming transcription explored and
deliberately **banked** (`plans/2026-07-22-concurrent-transcription.md`) — out of scope.

**Non-goals.** No new features. No streaming. No v2 AI-notes. **No crates.io publish** —
crates.io forbids git dependencies and both `cidre` (rev `a9587fa`) and `silero` (rev
`26a6460`) are pinned git revs, so distribution is clone-and-build only.

> Revised after `/audit-plan` (run `wf_0cbd5ead-5c2`, verdict `revise-major`). The v1 plan
> renamed the bundle-id; that rename was the sole source of both blockers the audit found
> and has been **cut** — see the decision below.

---

## The decision that shapes everything: the bundle-id stays

`com.lucianolupo.meetscribe` is documented across the repo as a **FROZEN cross-phase
invariant** — it binds the TCC grant, and changing it or the signing identity zeroes the
microphone + system-audio approvals.

v1 of this plan proposed renaming it to strip my name from a stranger's launchd label.
Two findings killed that:

1. **A rename cannot remove the name anyway.** The convention-correct form for this repo
   is `io.github.lucianolupo.meetscribe` (decided: personal namespace, no `meetscribe`
   org). `io.github.meetscribe` was rejected — that namespace maps to a real GitHub
   account and no such user or org exists. So the rename's stated benefit evaporates.
2. **The rename was the only destructive step in the release, and it was booby-trapped.**
   `LABEL` (`src/launchd.rs:13`) is the single source for the plist filename, the
   `launchctl bootout` target in `run_install` (:158) and `run_uninstall` (:220), the tray
   stop target (`tray.rs:314`), and `codesign --identifier`. Renaming it makes
   `run_install` boot out a job that no longer exists → the old agent keeps executing
   `~/.meetscribe/bin/meetscribe` while step 2 `fs::copy`s over that exact path (the
   comment at :156 exists precisely to prevent this) → the old agent still holds
   `~/.meetscribe/daemon.lock`, so the new job loses the flock (`daemon.rs:61-78`), exits
   nonzero, and `KeepAlive{SuccessfulExit:false}` crash-loops it forever. `uninstall`
   never removes the old plist, so `RunAtLoad` resurrects it every login.

**Decision: keep `com.lucianolupo.meetscribe` permanently.** Cost is a mild convention
violation (`com.<name>.*` conventionally implies an owned domain). Benefit is that the
release contains **zero destructive steps** — no `LEGACY_LABELS` migration code to carry
forever, no rehearsed rollback, no TCC re-approval, and both audit blockers cease to exist
rather than needing mitigation.

The "one-way door before publication" concern only binds if the id might change later. It
won't: **§2.5 re-asserts the freeze at the current value** with post-publication wording.

Only the **signing identity** actually blocks strangers, and unwelding it is
non-destructive — see Phase 2.

---

## Phase 1 — Legal, hygiene, and a CI baseline

Ordering matters: the notices need text that step 4 deletes, and CI must be green *before*
Phase 2 so a red run can't be confused with "Phase 2 broke it."

1. **`LICENSE`** — MIT, `Copyright (c) 2026 Luciano Lupo`. Chosen because `src/capture.rs`
   derives from MIT-licensed meetily; matching licenses keeps the notice chain trivial,
   and Apache-2.0's patent grant buys little on a personal audio tool.
2. **`THIRD-PARTY-NOTICES.md`** — **required, not optional.** `src/capture.rs` is a
   derivative work of meetily's `core_audio.rs`; deleting `docs/meetily-ref/` does not
   discharge the MIT notice obligation, and `capture.rs:7`'s header comment is necessary
   but not sufficient.
   - Fetch meetily's copyright line + MIT text **from upstream** —
     `gh api repos/Zackriya-Solutions/meetily/contents/LICENSE`. The vendored copies carry
     no header (`docs/meetily-ref/core_audio.rs` opens with a bare descriptive comment), so
     the text is not recoverable from what's being deleted.
   - Scope to the meetily derivation **only**. No binary is distributed, so no
     dependency-license bundle is required; when one is, generate it with `cargo about`
     from `Cargo.lock` rather than hand-listing crates that go stale on `cargo update`.
3. **`Cargo.toml` `[package]`** — `license = "MIT"`, `description`, `repository`,
   `authors`, `readme`, `keywords`, `categories`, `rust-version` (see Phase 3 for the
   verified MSRV).
4. **`.gitignore`: `/capture` → `/capture*`.** `capture2/` holds **3 real meeting WAVs**
   (17 MB, verified) and is shielded only by the global `*.wav` glob, unlike `capture/`
   which is ignored as a directory. One `transcript.md` or run log away from committable,
   and a private→public flip is irreversible for anything already pushed.
5. **`.github/workflows/ci.yml`** — `cargo check` + `clippy -D warnings` + `cargo test` on
   `macos-latest`, **full suite, no filtering**, caching `~/.cargo` + `target/` on the
   `Cargo.lock` hash. Get one green run on the private repo before Phase 2.
   - v1 of this plan claimed capture tests need hardware or TCC — **false**: the suite is
     **36 tests** (34 `#[test]` + 2 `#[tokio::test]` in `db.rs`), zero `#[ignore]`, and
     `capture.rs`'s two are pure (`gap_to_frames` arithmetic and `StartError` Display
     strings). Git deps fetch fine on Actions. Filtering on that phantom would ship a green
     badge proving nothing.
     - *(The `/audit-plan` run reported 34 by grepping only the literal `#[test]`; it
       missed the two `#[tokio::test]`. `cargo test` is the authority — 36.)*
   - Real cost is build-side (silero → ONNX Runtime, whisper.cpp + Metal, cidre's SDK
     floor). If the cold build proves slow, switch to `workflow_dispatch` + weekly
     `schedule` — do **not** narrow the test set.
6. **The vendored reference source is KEPT — user decision, 2026-07-30.** A prior revision
   proposed `git rm -r docs/meetily-ref docs/swift-ref`; **the deletion was declined.** The
   13 files stay, and the licensing obligation is discharged **with them in place**:
   - `THIRD-PARTY-NOTICES.md` covers **redistribution**, not merely derivation — muesli
     (`Copyright (c) 2026 Pranav Hari`) and pasrom (`Copyright (c) 2025 pasrom`) are named
     as verbatim vendored copies with full MIT text, alongside meetily. Copyright lines
     fetched from each upstream's `/license` endpoint.
   - `LICENSE` carries a **scope carve-out** stating that meetscribe's MIT grant does not
     relicense the two ref directories.
   - `docs/meetily-ref/README.md` and `docs/swift-ref/README.md` record upstream, license,
     copyright, why the files are kept, and the undated-snapshot caveat.
   - A copy is also archived at `~/Documents/Research/2026-07-18-granola-local-macos/refs/`
     with `PROVENANCE.md` (brain fact **#901**) — belt and braces, not a substitute.
   - The prose pointer at `plans/2026-07-18-phase1.5-model-provisioning.md:38` is **live
     again** and needs no rewrite.

**Verify:** `cargo build` green; every ref directory has a provenance `README.md`;
`THIRD-PARTY-NOTICES.md` names every vendored upstream with its own copyright line. CI
green. *(The old `git grep -n -E 'meetily-ref|swift-ref'` → empty gate is deleted — it was
only meaningful under the declined deletion.)*

---

## Phase 2 — Unweld the signing identity from my machine

The only thing that genuinely blocks a stranger: `codesign --sign 155971FE…` fails on any
machine but mine. **Non-destructive** — verified that `security find-identity -v -p
codesigning` returns exactly **one** identity here and its hash *is* the hardcoded one, so
the auto-detect tier resolves to the identical value and TCC cannot be disturbed.

1. **Delete the `SIGN_IDENTITY` const** (`src/launchd.rs:15`). Resolution order:
   1. `meetscribe install --identity <hash>`
   2. `$MEETSCRIBE_SIGN_IDENTITY`
   3. `security find-identity -v -p codesigning` — **only if exactly one** identity exists
   4. otherwise, an actionable error naming all three
   - Slots into the existing hand-rolled argv match at `launchd.rs:117-133` (one arm + one
     line of `-h` usage), and restores the repo's own documented precedence
     ("flag > config > default", `config.rs:5`).
2. **Do NOT add a `~/.meetscribe/config.toml` tier.** `config.rs:1-12` scopes that file as
   the *daemon's* runtime surface and the daemon never reads this key; the file is created
   only by `Config::load_or_init` at daemon startup (`config.rs:236-247`) while
   `run_install` creates `bin/models/logs/sessions` but not `config.toml` — so on the fresh
   clone this plan exists to serve, **the tier is structurally unreachable**. It would also
   force an `Option<String>` field, a `DEFAULT_CONFIG_TOML` entry, and a fix to
   `default_template_parses_to_default` (`config.rs:259-262`) — and a mistyped key falls
   into `#[serde(flatten)] extra` and silently falls through to auto-detect, signing with a
   different identity than configured. Hardest possible failure to diagnose.
3. **Keep** the `--remove-signature` → `--sign` order and the explicit `--identifier`
   (rustc's default id changes every build → would break TCC persistence).
4. **Rewrite the signing sections** of `docs/DAEMON.md` (:56-59, :80-83, :109-113) and
   `RESUME.md` to the new ladder. This is a **rewrite, not a find-replace** — the manual
   `codesign --sign <hash>` blocks no longer describe the supported path. These are the
   install instructions a stranger follows, so a stale hash there is a broken instruction,
   not a cosmetic blemish.
   - ⛔ **DO NOT "fix" the Team ID.** A previous revision of this plan called `L634X3YJBF`
     stale and proposed replacing it with `7C4H63K7S8` across 5 files. **That was wrong and
     would have corrupted a TCC-load-bearing value in the stranger-facing install doc.**
     The Team ID is the certificate's **OU**, not the parenthetical in its common name:
     `subject= CN=Apple Development: …(7C4H63K7S8), OU=L634X3YJBF`, and the live signed
     binary reports `TeamIdentifier=L634X3YJBF`. **The repo is correct.** Never write
     `7C4H63K7S8` anywhere labelled "Team ID".
   - Leave `plans/` verbatim — historical record. One dated superseded banner per affected
     file, no line edits.
5. **Re-assert the freeze** (do not delete it). Rewrite the FROZEN comments at
   `Info.plist:5-6` and `launchd.rs:14` to post-publication wording: *this id is permanent;
   changing it after publication zeroes TCC for every existing installation and would
   require a `LEGACY_LABELS` migration plus a documented re-approval step in the release
   notes.* The identity is no longer frozen; the **id** still is.
6. **Move `RESUME.md` → `docs/`** — a personal work log is the wrong thing to greet a
   visitor at repo root. Five tracked plan files reference the root path and
   `plans/2026-07-19-phase4-daemon.md:159` cites `RESUME.md:59` by line number; handle via
   the same dated banner rather than editing five historical files.

**Verify — two greps, not one:**
- `git grep -i 155971FEAE -- . ':!plans/'` → empty.
  **Denylist of the one exempt directory — NOT an allowlist of named paths.** Repo-wide is
  unsatisfiable and contradicts §2.4: 7 of the 11 `155971FEAE` hits live under `plans/`,
  a historical record kept verbatim under a dated banner, expected to retain the old hash.
  `L634X3YJBF` is **not** in this gate — it is a correct value that stays.
  - ⚠️ **This started life as an allowlist (`-- src/ docs/ Info.plist README.md
    CONTRIBUTING.md`) and that was a bug.** It named directories but not the repo root's
    other files, so `RESUME.md` — a personal work log carrying the hash and a now-broken
    `codesign --sign` recipe — sat at the root of the **already-public** repo while the gate
    reported PASS. A gate that enumerates what to check will always miss what nobody thought
    to enumerate. Exempt the known exception; scan everything else.
- `git grep -in lucianolupo -- . ':!Cargo.toml' ':!LICENSE' ':!THIRD-PARTY-NOTICES.md' ':!plans/'`
  → only legitimate `LucianoLupo/meetscribe` URLs. The exclusions matter: Phase 1.3
  legitimately adds the name to `Cargo.toml` `authors`/`repository`, and the bundle-id
  legitimately retains it everywhere else.
- `cargo build && cargo test && cargo clippy -D warnings`, then `meetscribe install` —
  **no TCC re-approval expected.** If a prompt appears, stop: something disturbed the
  signature.

---

## Phase 3 — README (the actual front door)

No README exists today. Lead with the privacy model — it's the entire pitch.

**Split by dependency.** These halves are *not* equally parallelizable:

**Parallel with Phase 1:**
- **What it is** — local-first background macOS meeting transcriber. No bots, no cloud, no
  telemetry, no runtime model download. Zoom / Meet / Teams / Slack huddles.
- **How it works** — one Core Audio aggregate on one clock; mic = ch0 "You", global process
  tap = ch1 "Others" → free channel-based speaker attribution, no neural diarizer.
- **Known limits, plainly** — multilingual `large-v3` default (meetings are Spanish);
  batch transcription at meeting-finalize, not streaming (link the banked plan); per-process
  taps fail on Teams so a system-wide mixdown tap is used
  ([pasrom#79](https://github.com/pasrom/meeting-transcriber/issues/79)); transcripts are
  **plaintext SQLite at 0600**, encryption-at-rest deferred to v1.1.
- **Credit** — meetily as the port source, up top, not buried.

**After Phase 2** (depends on the resolution contract Phase 2 invents, and links
`docs/DAEMON.md` which Phase 2.4 rewrites):
- **Install** — clone → `models/provision.sh` → build → sign → `meetscribe install` →
  approve two prompts.
- **Honest requirements** — Apple Silicon; **stable Rust ≥1.88** — *not* 1.85. Edition 2024
  stabilized in 1.85, but that is the edition floor, not the dependency floor: `cidre`,
  `home`, and the `time` crates each declare `rust-version = 1.88`, and cargo hard-errors
  below it. **Proven, not asserted:** `cargo +1.88 check --all-targets` finishes clean and
  `cargo +1.87` fails with *"rustc 1.87.0 is not supported by the following packages:
  cidre@0.11.3 requires rustc 1.88"*. (v1's "nightly-ish" was also wrong — no
  `rust-toolchain` file, no `#![feature]`, builds on stable.) Then **macOS 14.4+**, not
  14.2 — per this project's own
  `docs/research-2026-07-18.md:20,30,46` ("API since 14.2; ship ≥14.4 for correct TCC
  category"); **your own Apple Development certificate**, with the why (TCC binds to the
  signature); ~4 GiB disk for models.

**Verify — the piece v1 was missing entirely:** fresh `git clone` into a temp dir with an
empty `models/`, follow the README **verbatim** through `provision.sh` + `cargo build` +
the sign step. **Stop before `meetscribe install`** to avoid colliding with the live daemon.
This is the only thing in the plan that exercises `provision.sh` against an empty `models/`
(4.1 GiB, shas scraped from HuggingFace `/raw/main` git-lfs pointers — one upstream layout
change and step 2 of the install dies) and a genuinely cold build.

⚠️ **It does NOT cover the identity ladder** — an earlier revision claimed it did, wrongly.
`SIGN_IDENTITY` is read only inside `codesign()` (`launchd.rs:30-48`), called only from
`run_install`, which this verify deliberately skips. And skipping is correct: `run_install`
step 1 boots out `gui/$UID/com.lucianolupo.meetscribe`, which is **label-scoped, not
HOME-scoped**, so running it in a throwaway clone would knock out the live daemon anyway.
Cover the ladder with **unit tests instead** (Phase 2.1): split pure
`parse_identities(&str) -> Vec<String>` and `resolve_identity(flag, env, found)` functions
out of the shell-out, and test zero / exactly-one / multiple. Those branches are
unreachable on this machine, which has exactly one identity.

---

## Phase 4 — Release hygiene

1. **`CONTRIBUTING.md`** — contributor process **only** (test + clippy gate, commit/PR
   flow). **Link** `docs/DAEMON.md` for the build/sign/install loop; do not restate it. One
   owner per fact — duplicating the recipe recreates exactly the drift this plan cleans up.
2. **GitHub repo settings** — description, topics, Issues on.
3. **Flip private → public.** Explicit preconditions, all must pass:
   - both Phase-2 greps clean
   - `git ls-files | grep -iE '\.wav$|^capture'` → empty
   - `git status --untracked-files=all` → clean
   - CI green
   - Phase 3's clean-clone verify passed
   - the `Recording→Transcribing→Idle` transition observed on a real call (see Loose ends)

---

## Sequencing

```
Phase 1 (LICENSE+notices → ref-dir READMEs → Cargo.toml → gitignore → CI)
   ↓
INTEGRATION: commit → git push -u origin chore/oss-release-prep → open PR
             (a PR is what actually fires CI — `push:` is scoped to master,
              so CI has never run on this branch) → green → merge to master
   ├── Phase 3 "what it is / how it works / limits / credit"   [parallel]
   ↓
Phase 2 (signing identity; non-destructive, no meeting-free window needed)
   ↓
Phase 3 "install / honest requirements" + clean-clone verify (from master)
   ↓
Phase 4 → flip public
```

**Commit before running any verify grep** — an uncommitted tree makes `git grep` results
depend on staging state.

v1's blanket "Phase 1 → Phase 3 can proceed in parallel" was false for the half that
matters. **No phase is destructive**, so no step needs a meeting-free window.

## Open questions

1. **Do `plans/` and `docs/research-2026-07-18.md` ship publicly?** Recommending **yes** —
   the 96-source research and the `/audit-plan` findings are a genuine differentiator and
   contain no secrets. Note it makes the dated superseded banners publicly visible; the
   Phase-1 content grep catches the dead pointers.
2. **Repo name/scope** — ship as `LucianoLupo/meetscribe`. Recommending as-is.

*(v1's question #1 — the bundle-id — is closed: it stays, see the decision section.)*

## Loose ends

- **`Recording→Transcribing→Idle` has never been observed on a live call** with the current
  binary — the one unverified piece of the `ed37c80` fix, and the transition a menu-bar user
  stares at. Since this plan no longer forces a reinstall, it needs one deliberate real
  call. **Promoted to a precondition on the public flip** — much worse to learn from a
  stranger's day-2 issue.
- Brain todo **#37** (`status.json` stuck at `"recording"`) is **stale** — fixed in
  `ed37c80`, merged in PR #1. Mark done.
- Memory topic file + `MEMORY.md` still say `origin/master = 91ece0e`; actual is `086c943`.
- `~/.meetscribe/config.toml` on this machine (Jul 22) predates any template change; the
  `DEFAULT_CONFIG_TOML` edits never reach existing installs. Nothing in this plan changes
  that file's contents, but note it if a future change does.

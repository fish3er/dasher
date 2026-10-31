# Phase 0 — diagnosis for `beamserach-updated`

Date: 2026-10-27. Branch `beamserach-updated` = `gemma4-beamserach` @ `ec315e1` (no commits on top).
No production code changed. New files: `tools/inspect_tokenizer.py`, `tools/diag_phase0.py`, this note,
`notes/diag/*.json`.

Raw data:
- tokenizer: `notes/diag/tokenizer_stats.json` (`flatpak-spawn --host python3 tools/inspect_tokenizer.py --out …`)
- model runs (a)+(d): `notes/diag/phase0_model.json` (`flatpak-spawn --host python3 tools/diag_phase0.py --out …`)
- (f) was run in the sandbox (pyphen 0.18.1 wheel in the scratchpad, not installed); numbers are quoted inline.

Environment: host Python 3.14.8, llama-cpp-python 0.3.28 (built locally 2026-06-11, Vulkan, RX 6800 XT),
model `models/google_gemma-4-E4B-it-Q4_K_M.gguf` (42 blocks, `sliding_window=512`, 18 shared-KV layers,
`final_logit_softcapping=30`, n_vocab 262 144).

---

## a) Current `BeamSearch`

**Interface.** `BeamSearch(gguf_path, n_gpu_layers=-1, n_batch=512, n_ctx=2048, seed=None)`;
`suggest(prefix: str, n=5, beam_width=5, top_k=None, top_p=1.0) -> list[Suggestion(text, score, level, complete)]`.
`level` is `mid_word` if the prefix doesn't end with whitespace, otherwise `word_boundary` (the prefix is
`rstrip()`ed before tokenization). `text` is only the continuation (`ają`, not `badają`). Score = cumulative
logprob / number of tokens. At most `_MAX_NEW_TOKENS = 6` steps.

**Callers.**
- `eval.py` → `BeamSearch.suggest` (uncached).
- `sweep.py` → same.
- `eval_context.py` → `context_sweep.CachedBeamSearch.suggest_tokens(prefix_tokens, level, …)` (subclass
  with prefix KV cache).
- **The UI does not call it.** `main.py` imports `model.Autocomplete` (HF transformers, `google/gemma-3-1b-it`,
  `num_beams=8`) — issue P9, still open on this branch.

**Where the prefix KV cache lives.** Only in `context_sweep.CachedBeamSearch`: the prefix is prefilled into
`seq 0`; `multi_seq` (default) does `llama_memory_seq_cp(0 → 1..B)` every step and replays beam tails,
`sequential` does `seq_rm(0, n_prefix, -1)` and replays each tail in `seq 0`. `_prefill` does a full
`llama_memory_clear` (the `kv_unified` determinism trap from CLAUDE.md).

**`_decode_last` / `_decode_batch` (base class).** `_decode_batch` **clears the whole KV and re-encodes the full
prefix for every beam**, each beam with **its own `seq_id`** (0..B-1) in one `llama_decode`, logits only at the
last position of each sequence. `_decode_last` splits the sequence list into chunks under two limits:
`n_seq_max` (patched to 64, `kv_unified=True`) and the `n_batch` token budget.

> ⚠️ **Correction to the plan's premise.** The plan describes the existing decoding as "sequential `seq_id=0`
> because of Gemma 4's interleaved SWA". That is not what the code does: both `BeamSearch._decode_batch`
> and the default `CachedBeamSearch` mode are **multi-sequence** (`seq_id` 0..B), and CLAUDE.md records
> `multi_seq` and `sequential` as producing matching results. Nothing in the repo forces `seq_id=0`.
> Phase 1 can use either; I'll measure both as the plan asks.

**Multi-token decode of ONE sequence with logits at every position — works.** Test: prefix in `seq 0`, then
the tail (10 or 13 tokens) in **one** `llama_decode` with `logits=1` everywhere, vs the same tail token by
token on the same cached prefix (`phase0_model.json → multi_position`):

| case | prefix tok | tail tok | one call | token by token | top-1 equal at every position | max KL (nats) | max abs Δlogprob, top-50 |
|---|---|---|---|---|---|---|---|
| short | 7 | 10 | **61.0 ms** | 195.9 ms | yes | 0.045 | 1.11 |
| prefix > SWA window (899 tok) | 899 | 13 | **54.4 ms** | 141.5 ms | yes | — | (max abs Δlogprob, full vocab 1.23) |

- No crash, no error. Positions past the 512-token SWA window behave like the short case.
- **The results are not bit-identical to token-by-token decoding.** Per-position KL is 0.00–0.045 nats, and the
  largest logprob deltas are on low-probability tokens. The difference is deterministic: repeating the same
  one-call decode gives max |Δ| = 0.0. It depends on batch shape (n tokens > 1 versus 1). My guess is the
  Vulkan matmul kernel path; I haven't verified that. **Consequence for Phase 2:** one-pass path scoring is
  usable, but levels scored in one pass and branches scored by separate calls will differ by up to about 1
  logprob on tail tokens. Sibling normalization and gating thresholds have to tolerate that. I'll report it
  next to the latency comparison and won't call the two methods "equivalent".
- The one-call path is 2.6–3.2× faster than token-by-token for 10–13 tokens.

## b) UI (`main.py` on this branch)

**Drawing.** One `tk.Canvas` 1000×900, redrawn from scratch (`delete("all")`) every 16 ms by `run()` →
`update_physics()` + `draw_scene()`. `self.elements` is a **flat** list of `{label, low, high, color, is_ai}` in
[0, 1] coordinates; drawn as rectangles whose left edge moves right as the box gets smaller
(`x_left = cross_x + (1 - h/canvas_h) * (canvas_w - cross_x)`). There's no nesting: children of a box are
invisible until it's entered.

**Layout.** `update_layout()`: the top **30% is always reserved** for AI suggestions (`ai_ratio = 0.3`, equal
heights), the remaining 70% holds the alphabet with **uniform** heights. If there are no suggestions the 30%
stays empty, so the letters never move when suggestions arrive. Alphabet = `aąbcćdeęfghijklłmnńoóprstuwvyzźż` +
`' ' . , ! ?`: **lowercase only** (the UI cannot type capitals), and **no `q` / `x`**.

**Steering.** `dx = (mouse_x - cross_x) / (canvas_w - cross_x)` → zoom speed `dx * 0.08` (left of
`cross_x = 80` zooms out). `dy = sign·|dy_raw|^1.5`, where `dy_raw` is the mouse offset from `mid_y`, pans the
view by `dy * 0.06 * range`. Zoom is centered on the **view center** (`mid_y`).
**Selection is not a line crossing.** A box is written when the visible interval lies entirely inside it
(`view_min >= low and view_max <= high`), i.e. when it fills the screen height. The coordinates are then
remapped into the box (`(view - low) / span`) and a new layout is built. Zooming out past
`range > 1.05` pops `history` (undo).

**When predictions refresh / threads.** After every selection a new daemon `threading.Thread` runs
`ac.predict(typed_text, 5)` and hands the result back with `root.after(0, apply_suggestions)`. There is
**no cancellation and no staleness check**: a slow result for an old `typed_text` can overwrite a newer one.
The suggestion boxes appear in the top 30% while the user is already flying.

**Context given to the model.** The whole `typed_text` (the session so far), tokenized with HF
`truncation=True, max_length=128`. HF truncates from the **right** by default, so once the text passes about
128 tokens the model gets the **beginning** of the session and loses the end. There's no notion of "previous
sentence".

**Bugs found while reading (outside the plan's scope; not fixed):**
1. A mid-word AI label is `word_prefix + suffix` (the full word), and `check_selection` appends it to
   `typed_text`. `bad` + `badający` → `badbadający`.
2. The 128-token right-truncation described above.
3. The staleness race described above.

**Proposal for the three bands without changing the steering model.**
- Keep the coordinate system and the "viewport inside box → write it and remap" rule as they are. That rule
  already implements recursion: after entering, the box's interior becomes the new [0, 1].
- Replace the flat `elements` list with a **node tree**: `Node(label, low, high, kind∈{phrase,tree,letter},
  children_lazy)`. The root interval is split top to bottom into `phrases [0, a)`, `tree [a, b)` and
  `letters [b, 1]`, with `1 - b >= letters_min_share` (config, 0.25). When a band is empty, `a` or `b`
  collapses.
- `draw_scene` recurses into a node's children (in the node's local coordinates) once the node is taller than
  a threshold in pixels. That makes L2/L3 visible before they're entered. When a node is entered, its
  precomputed children are the new top-level layout: L2 sit in the tree band, and the node also gets its own
  phrases and letters bands.
- Freezing order: a node's children get their `low/high` once, when they're created. Asynchronous results only
  fill nodes that are still empty, or animate height changes with interpolation; they never reorder siblings
  that already exist. This also fixes the staleness race, because results are attached to the node they were
  computed for, not to "the current text".
- Note that P9 is fixed as part of this: the UI moves to the Gemma 4 GGUF backend (`--search gv`).

## c) Gemma 4 tokenizer (`notes/diag/tokenizer_stats.json`)

| quantity | value |
|---|---|
| n_vocab / control / byte-fallback `<0x..>` / normal | 262 144 / 18 / **256** / 261 870 |
| tokens with a space anywhere other than position 0 | **31**: 30 are pure multi-space runs, 1 is `>▁</` (HTML) |
| …of those, containing letters (a word crossing a boundary) | **0** |
| …ending with a space after text | **0** |
| pure-space tokens | 31 (lengths 1…31, one of each) |
| tokens containing `\n` | 31, **none mixed with text** |
| tokens containing `\t` | 31 |
| `valid_vocab` (UI alphabet + space, case-insensitive) | **135 435**, of which 84 139 start with a space |

- **The assumption "spaces only at token start" holds for letter text.** The only exceptions are pure-whitespace
  runs and `>▁</`. Neither can occur inside a word, so the cut can be `rfind(' ')`, and no fallback to a token
  boundary is needed. One caveat: the 30 multi-space tokens are in `valid_vocab`, because every character is a
  space. I propose dropping tokens with more than one consecutive space from the index, since the UI produces
  single spaces.
- **Diacritics: every one of ą ć ę ł ń ó ś ź ż, lowercase and uppercase, has its own single token.** Single-character
  tokenization never uses byte fallback for Polish, so byte-fallback tokens can be left out of the index. Some
  tokens with a leading space are missing: ` ą`, ` ę`, ` ń`, ` Ą`, ` Ę`, ` Ń`, ` Ź`. They're harmless, because a
  space-prefixed letter can still be produced as ` ` + letter with two tokens.
- No dummy prefix: the first word of the text tokenizes without `▁` (`nau`, `id`).

Tokenizations:

| text | tokens |
|---|---|
| `naukowcy bad` | `nau` `kow` `cy` `▁bad` |
| `naukowcy badający` | `nau` `kow` `cy` `▁bad` `ający` |
| `idę d` | `id` `ę` `▁d` |
| `idę do domu` | `id` `ę` `▁do` `▁domu` |
| `źdźbło` | `ź` `d` `ź` `b` `ło` |

`idę d` → `▁d` versus `idę do` → `▁do`: this is the case where re-tokenization matters. The prefix forces `▁d`,
but the canonical tokenization never writes `do` as `▁d` + `o`.

## d) Problem reproduction (`phase0_model.json → beam_search, char_dist`)

Current `BeamSearch.suggest`, uncached, 5 warm reps:

| prefix | config | suggestions (text, score) | target | rank | latency mean / p50 |
|---|---|---|---|---|---|
| `naukowcy bad` | bw5 k16 n5 (current default) | `ają` −1.03 | `ający` | — | 84 / 77 ms |
| `naukowcy bad` | bw10 k16 n10 | `ają` −1.04, `ani` −1.77 | `ający` | — | 103 / 104 ms |
| `idę d` | bw5 k16 n5 | `anke` −1.59, `awning` −2.01 | `o` | — | 77 / 75 ms |
| `idę d` | bw10 k16 n10 | `anke` −1.59, `awning` −2.01 | `o` | — | 93 / 93 ms |

I couldn't get a top-10: asking for n=10 returns **1–2** suggestions. For `idę d` they're English/German
(`danke`, `dawning`).

Next-character distribution (only the top entries; see the JSON):

| prefix | naive (no back-off; first character of the next token) | 1-token back-off (cut before the word, tokens starting with `rem`) |
|---|---|---|
| `naukowcy bad` | P(`a`) = **1.000** | P(`a`) = 0.9995 |
| `idę d` | `.` 0.604, `a` 0.270, ` ` 0.043, …, P(`o`) = **0.003** | `i` 0.643, `u` 0.183, `e` 0.110, `a` 0.025, P(`o`) = **0.018** |

**Hypothesis verdict: partly confirmed.**
- **`idę d` — confirmed as a mechanism.** The forced `▁d` gives P(`o`) = 0.003 and the most likely next character
  is `.`. The legacy beam can only continue `▁d…` and drifts into English. Backing off before the word raises
  P(`o`) about 6× (0.003 → 0.018). That's only a 1-token approximation, though, and it's still low for a second
  reason: **with no context the model doesn't expect `do` after `idę` at all.** After BOS+`idę` the next token is
  `,` with p = 0.92. After `Jutro po pracy idę` it's ` mag` 0.69, ` sa` 0.15, ` gym` 0.04, and ` do` isn't in the
  top 12. Back-off alone won't fix short words; **the previous-sentence context (Phase 1 note) is a precondition.**
  The `idę d` test in `tests/test_gv_search.py` needs a realistic context, or it will test the model's
  context-free prior and not the algorithm.
- **`naukowcy bad` — not this mechanism.** `▁bad` is the canonical token, and P(`a`) is already 1.0 without
  back-off. Legacy fails here differently: it returns a **single** suggestion, `ają`, which is complete because
  the next token was a space, and `ający` is never listed. Beams are lost to completion and dedup (the P2 family).
  I haven't diagnosed the exact cause, since `legacy` stays frozen for comparisons anyway.

## e) MTP drafter feasibility (report only; nothing downloaded or rebuilt)

- **Vendored llama.cpp:** llama-cpp-python v0.3.28 pins `vendor/llama.cpp` @ **`9e3b928`**
  ("common : relax sampler name matching (#23744)"), **2026-06-07 20:48 UTC**. Our wheel was built locally
  from the sdist on 2026-06-11 (`~/.cache/pip/wheels/…/llama_cpp_python-0.3.28-…whl`). The commit hash isn't
  embedded in the `.so`; I took it from the v0.3.28 tag on GitHub.
- **PR #23398 (Gemma 4 MTP),** merge `04eb4c4`, 2026-06-07 12:50 UTC: **included** (GitHub compare: `9e3b928` is
  4 commits ahead of it). The `.so` has the `gemma4-assistant` arch, `llama_get_ctx_other`, and
  `*_embeddings_nextn*`.
- **PR #24282 (Gemma 4 E2B/E4B assistants),** merge `7d2b45b`, 2026-06-08 20:48 UTC: **not included** (`9e3b928` is
  15 commits behind it). The unsloth drafter README says builds before #24282 can't load the E4B drafter.
- **Python API:** the ctypes layer exposes `llama_context_params.ctx_type` (`LLAMA_CONTEXT_TYPE_MTP = 1`) and
  `ctx_other`. There's **no high-level MTP or draft-model support**: `llama_speculative.py` only has
  `LlamaPromptLookupDecoding`. Even after a bump, the draft/verify loop would have to be written by hand over
  the low-level API.
- **What it would take:** either (1) bump `vendor/llama.cpp` to ≥ `7d2b45b`, rebuild with
  `CMAKE_ARGS="-DGGML_VULKAN=on"`, and check the ctypes signatures against the new `llama.h` (the bindings lag
  behind the C API, so a bump past their pin can break struct layouts); or (2) run a **`llama-server` sidecar**
  built from current llama.cpp with Vulkan, using `--spec-type draft-mtp --model-draft …`, for measurements only.
  (2) is much lower risk for a one-session spike. The README's `-fa off` note concerns a CUDA flash-attention
  abort, and nobody has tested the Vulkan behaviour.
- **Drafter GGUF:** `unsloth/gemma-4-E4B-it-GGUF`, folder `MTP/`: `mtp-gemma-4-E4B-it-Q8_0.gguf` **98.7 MB**
  (98 653 248 B, the recommended one, also at the repo root as `mtp-gemma-4-E4B-it.gguf`), plus `-F16` and `-BF16`
  at **171.8 MB** each.
- As the plan says, the drafter isn't a standalone LM: it reads the target's hidden state and KV (`ctx_other`).

**Verdict:** possible, but not with the current build. Phase 5 needs either a sidecar (my recommendation) or a
vendor bump and rebuild.

## f) Syllabifier

- No syllabifier is installed on the host or in the sandbox, and the host has `hyphen-en` only, no `hyphen-pl`.
- **`pyphen` 0.18.1** (pure Python, ships `hyph_pl_PL.dic`) works. It was tested from the scratchpad without
  installing it:

| word | default (left=2, right=2) | left=1, right=1 | correct syllables |
|---|---|---|---|
| badający | ba-da-ją-cy | ba-da-ją-cy | ba-da-ją-cy ✔ |
| naukowcy | na-ukow-cy | na-ukow-cy | nau-kow-cy ✘ |
| źdźbło | źdźbło | źdźbło | źdźbło (1 syllable) ✔ |
| przedszkole | przed-szko-le | przed-szko-le | przed-szko-le ✔ |
| pszczoły | psz-czo-ły | psz-czo-ły | pszczo-ły ✘ |
| idę | idę | i-dę | i-dę |
| opublikowali | opu-bli-ko-wa-li | o-pu-bli-ko-wa-li | o-pu-bli-ko-wa-li |

These are **hyphenation** patterns, not syllabification. They're morphologically sensible and good enough for
tile cuts, but they're wrong at diphthongs and consonant clusters (`na-ukow`, `psz-czo`). They need
`left=1, right=1` to split short words. Usable for 4.2 as an option. Using it requires adding `pyphen` to
`requirements.txt` and installing it on the host (`pip install --user pyphen`). I haven't done either.

---

## Decisions I need before Phase 1

1. **Normalization for Phase 4.** `eval_context.py` has **no normalization function to import**. Corpora are
   read raw (`path.read_text()` in `context_sweep.load_documents`). The only normalization in the repo is
   `" ".join(raw.split())` inside `eval.parse_sentences`. Options: (i) add `normalize_text()` to
   `eval_context.py`, define it as what the corpora get today (i.e. nothing), and use it from
   `prep_podcast.py`; or (ii) define real normalization (NFC + whitespace) and apply it to Lalka as well, which
   changes Lalka's inputs compared with the previous runs. I suggest (i), plus NFC inside `prep_podcast.py`
   only.
2. **Capitalization.** The UI is lowercase-only. I'll follow the paper: condition on lowercase, case-insensitive
   index, and capitalize sentence starts in the display only. Eval stays case-insensitive as before.
3. **Multi-space tokens in `valid_vocab`.** I'll exclude them unless you object.
4. **`pyphen` dependency.** Install it on the host and add it to `requirements.txt`, yes or no? If no, 4.2 is
   skipped.
5. **Phase 5 route.** If you approve the spike later: sidecar (recommended) or vendor bump.
6. **`main.py` was "NIE RUSZAĆ" in CLAUDE.md.** Phase 3 rewrites it, and the P9 swap to Gemma 4 GGUF comes
   with that. I'm treating the plan as lifting that rule; tell me if not.

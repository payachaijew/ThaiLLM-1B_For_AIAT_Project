# Reviewer Pre-Mortem

**Date:** 2026-08-18
**Method:** assume the paper was submitted to iSAI-NLP 2026 and **rejected**. Work backwards
from the rejection review to find what killed it, while there is still time to prevent it.

Each item states the reviewer's objection in the reviewer's own register, the honest assessment
of whether the objection lands, and the concrete defence — or, where there is none, the
admission that there is none.

---

## R1. "This is first-in-Thai dressed up as a controlled study."

**Does it land?** Only if we let it. The submitted framing decides this entirely.

AttnRes is already running in production frontier systems (Kimi K3, `arXiv:2607.24653`) and in a
multilingual 314B MoE (Motif 3, `arXiv:2608.09119`). Nobody is impressed that it now also runs
on Thai. If the abstract's first sentence contains "first Thai", the paper is dead on arrival.

**Defence.** The claim must be about the *regime*, not the *language*. Thai is the instrument;
compute-limited continued pretraining is the subject. Concretely: every paper in the matrix
trains from scratch or does general-domain mid-training with abundant compute; none asks what
happens when routed residuals must pay for their 12–45 % throughput cost out of a fixed
wall-clock budget while also not destroying the base model's existing languages. That question
is regime-specific and language-agnostic in principle — Thai is simply the case we can run.

**Residual risk:** Medium. Requires disciplined writing throughout, not just in the abstract.

---

## R2. "The gains come from extra parameters, extra data, or extra compute."

**Does it land?** Not on parameters, partly on compute.

- *Parameters:* MHAR adds **zero** parameters (`arXiv:2607.27230`). Delta's router adds one
  `d`-dimensional query vector plus a scalar per layer — for a 1B model with 16 layers that is
  roughly 3×10⁴ parameters, about 0.003 % of the model. A parameter-matched control is available
  and cheap, and `plans/research_plan.md` §7 already requires one if router parameters are
  material. They are not material. This objection is answerable with a table.
- *Data:* the plan already mandates identical documents, identical order, identical observed
  token counts across conditions. Answerable, provided the manifest hash is actually recorded.
- *Compute:* **this one lands.** See R3 and R8.

**Defence.** Report parameter counts to the digit, publish the manifest hash, and report both
axes. Do not report a token-matched win without the GPU-hour-matched number beside it.

**Residual risk:** Low on parameters/data, high on compute.

---

## R3. "The Standard baseline was weakened."

**Does it land?** **Yes — and worse than the authors realise.** This is currently the single
most dangerous objection, because right now the bias runs the *other* way and is invisible in
the plan.

`plans/research_plan.md` §7 correctly requires "optimized native Standard path, not a routed
wrapper with routing disabled". Good. But the local audit found three defects that all
handicap the **routed** arms, not the baseline:

- **E4:** `HFDepthRouterAdapter` hard-disables gradient checkpointing because its routing state
  is not re-entrant. S0 can use gradient checkpointing; D1 and D2 cannot. That is a different
  memory/throughput regime per condition — the GPU-hour axis is measuring the wrapper, not the
  architecture.
- **E2:** the routed arms retain the autograd graph in `last_routing` after every forward,
  pinning activation memory that the baseline never pays for.
- **§5.5:** the `route_scale` zero-gate makes routing-parameter gradients *identically zero* at
  initialisation and ~0.1 % of the paper-faithful magnitude after 50 steps.

So the realistic failure mode is not a weakened baseline — it is a **crippled treatment**
producing a false negative that we then publish as "routing does not help Thai CPT". That is a
worse outcome than rejection: it is a wrong result in the record.

**Defence.** All three must be fixed and re-validated before any pilot. Report throughput and
peak VRAM for all three conditions under *identical* checkpointing and memory settings, or
declare the difference explicitly and treat the GPU-hour axis as untrustworthy.

**Residual risk:** Currently **critical**. Reducible to low, but only by doing the work.

---

## R4. "Token-matched and GPU-hour-matched give different answers — so which is it?"

**Does it land?** It is not an objection, it is the paper. But it must be pre-registered as the
expected outcome, not discovered afterwards and spun.

MHAR reports that even *with* fused Triton kernels, AttnRes-family training runs at
**0.55–0.88×** baseline throughput. Take the favourable end: at 0.88×, an equal-wall-clock
budget buys the routed arm 12 % fewer tokens. The promotion threshold in
`experiment_parameters.json` is a 2 % relative Thai BPB improvement. A 12 % token deficit
plausibly costs more than 2 % BPB. At the unfavourable end (0.55×), the routed arm sees 45 %
fewer tokens and the comparison is not close.

**The honest prediction is therefore: routed variants win per token and lose per GPU-hour.**

**Defence.** Say so *in advance*, in the preregistration, and frame the paper around it: "for a
Thai lab with a fixed GPU budget, does architectural routing buy anything?" — with a
well-supported "no, at this scale, under these kernels" being a genuinely useful answer for the
Thai research community. Pre-registering the prediction converts the most likely outcome from
an embarrassment into the contribution. Failing to pre-register it converts the same outcome
into an accusation of post-hoc framing.

**Residual risk:** Low **if** pre-registered. High if not.

---

## R5. "There is newer work you did not cite."

**Does it land?** It nearly did already. The registry as audited was missing seven relevant
works, two of them from the past three weeks (`arXiv:2608.01075`, 2026-08-02;
`arXiv:2608.09119`, 2026-08-10), and cited one paper under a title its authors abandoned in v2
(`arXiv:2606.06564`, WAV → HAARES).

The field is producing roughly one relevant paper every ten days. Between this audit and the
2026-09-01 deadline, expect one to two more.

**Defence.** The registry is now updated. Re-run queries Q1–Q4 and T4 within 48 hours of
submission and log the result. Cite HAARES by its v2 title.

**Residual risk:** Medium, irreducible — this is a fast-moving field and it is a coin flip
whether something lands in the next fortnight.

---

## R6. "Your result just confirms the Delta paper. Where is the new knowledge?"

**Does it land?** Yes, if the result is a plain replication of the Delta ranking.

If the finding is "Delta > Standard, and MHAR > Delta, in Thai too, at the same relative
magnitudes", the reviewer's response is that the Delta and MHAR papers already established the
ranking across 220M–8B and the Thai instantiation adds no knowledge.

**Defence.** Three things make a confirmation publishable, and at least one must hold:
1. The ranking transfers per token but **inverts** per GPU-hour (the predicted outcome, R4).
2. The magnitude is systematically different in CPT than in from-scratch training — e.g. routing
   helps much less when starting from a converged checkpoint, which would be a real statement
   about conversion rather than about the architecture.
3. Acquisition and retention **dissociate** — routing helps Thai and hurts English, or the
   reverse. Nothing in the matrix measures retention at all; the whole column is empty. Any
   solid finding here is new.

If none of the three hold, the honest move is to not submit this as a research paper.

**Residual risk:** Medium.

---

## R7. "What is a negative result worth here?"

Substantial, and this is worth being explicit about because it changes the risk profile of the
whole project.

A well-controlled "routed residuals do not pay for themselves in Thai compute-limited CPT at
~1B" tells every Thai and SEA lab planning a CPT budget to spend it on data and tokens rather
than on architectural routing. That is directly actionable for the audience physically in the
room at iSAI-NLP in Bangkok. It is also the kind of result that only a compute-limited lab has
any incentive to produce — frontier labs will never publish it.

**But it is only worth something if the treatment was implemented faithfully.** A negative
result from a crippled implementation (R3) is worthless and actively harmful. The value of the
negative result is entirely contingent on closing §5.5, E2 and E4 first.

---

## R8. "The scope is too large for three people and 4×A100."

**Does it land?** **Yes. Decisively. This is what actually kills the project, not novelty.**

Today is 2026-08-18. The extended deadline is **2026-09-01**: **14 days**.

Remaining work in the current plan, in dependency order:

| # | Task | Realistic time |
|---|---|---|
| 1 | Fix `route_scale` confound, E2 graph retention, E4 checkpointing; re-validate Tier A | 3–5 d |
| 2 | Implement MHAR / D2 from scratch (**no code exists**) and validate it | 3–5 d |
| 3 | Base-model screen across 5 candidates (tokenizer, frozen BPB, license, port) | 2–4 d |
| 4 | Data provenance + license + decontamination audit | 3–5 d |
| 5 | One-A100 preflight, 200 steps × 3 conditions | 1–2 d + queue |
| 6 | Premise pilot, 50–100M tokens × 3 conditions | 3–7 d GPU |
| 7 | Second seed for promoted conditions (mandatory before promotion) | ×2 on step 6 |
| 8 | Full 1B CPT × 3 conditions | weeks |
| 9 | Evaluation, statistics, write-up | 5–7 d |

Items 1–4 alone exceed the calendar, and they are all pure prerequisites — none of them
produces a single number that goes in the paper. Steps 6–8 have not started, no GPU allocation
is confirmed, and the sequence is strictly serial (the pilot gates the full run).

**There is no defence.** The 1B study in `plans/research_plan.md` cannot be delivered by
2026-09-01. Any plan that assumes otherwise is not a plan.

**Implication.** The decision facing the team is not "is this topic novel enough" — the audit
says it is marginally but genuinely novel. It is "what can honestly be finished in 14 days".
See `decision_memo_th.md` §3 for the two options and the go/no-go date.

---

## R9. "Your mechanism analysis is already published."

**Does it land?** Yes, largely.

`arXiv:2606.13168` (2026-06-11) performs causal routing-ablation probes on Block AttnRes, using
**Qwen3-0.6B** — the exact model `plans/pre_train_validation_plan.md` designates as the pilot —
and reports that routing mass dissociates from causal importance. The archived harness's
clamping/ablation/override machinery (`set_next_intervention`) implements essentially the same
class of intervention.

**Defence.** Demote mechanism analysis from a headline contribution to a supporting result, cite
NW08 prominently as prior art, and narrow our version to the one question it does not answer:
whether routing distributions differ systematically across **Thai, English and controlled
code-switch** input. Framing it as "we extend NW08's finding to the cross-lingual case" is
honest and still worth a section. Framing it as our own novel mechanism analysis invites a
plagiarism-adjacent objection.

**Residual risk:** Low once reframed; high if the reframing is not done.

---

## R10. "You claim a 1B study but ran 0.6B."

**Does it land?** It would be fatal, and it is a live risk *right now* because of the title.

The working title is *"...A Controlled 1B Study"*. The only computationally feasible near-term
experiment is the Qwen3-0.6B pilot. If the paper ships under a 1B title with 0.6B results, that
is a misrepresentation, and at an in-person regional venue where the authors will be standing
next to the poster, it is also a reputational risk.

**Defence.** Change the title the moment the scope changes. If the experiment is 0.6B, the title
says 0.6B. If the pilot is the paper, the title says "a pilot study". Sub-1B is not a weakness
at iSAI-NLP — misdescribing it is.

**Residual risk:** Low, but only because it is entirely within the team's control. It requires
someone to actually do it.

---

## Summary of pre-mortem outcomes

| ID | Objection | Lands? | Residual risk after defence | Fixable in 14 days? |
|---|---|---|---|---|
| R1 | Only first-in-Thai | If badly framed | Medium | Yes — writing only |
| R2 | Gains from params/data/compute | Partly (compute) | Low / High | Yes |
| R3 | Baseline weakened → actually **treatment crippled** | **Yes** | **Critical** | Partly (3–5 d) |
| R4 | Token vs GPU-hour disagree | Is the paper | Low if pre-registered | Yes |
| R5 | Missing newer work | Nearly did | Medium, irreducible | Yes |
| R6 | Only confirms Delta | Yes | Medium | Depends on result |
| R7 | Negative-result value | Favourable | — | Contingent on R3 |
| R8 | **Scope too large** | **Yes, decisively** | **Fatal for 1B** | **No** |
| R9 | Mechanism already published | Yes | Low once reframed | Yes |
| R10 | 1B title, 0.6B result | Would be fatal | Low | Yes — rename |

**The two that decide the project are R3 and R8.** R8 is not fixable; the scope must change to
match the calendar. R3 is fixable but must be fixed first, because every downstream number —
including the negative result that gives the project its fallback value — is meaningless until
it is.

You made a wrong prediction on the paper below. Your task is to reason carefully over what went wrong and update your Meta Review — a personal set of lessons that will help you judge future papers more accurately.

---

## What you are given

- The full paper (anonymized), including all figures, tables, and proofs.
- Your original prediction and reasoning.
- The correct label (ACCEPT or REJECT).
- Your current Meta Review (may be empty on first error).

---

## How to diagnose your error

Work through these questions honestly:

1. **What did I predict, and what was the correct answer?**

2. **What was my reasoning?** Reconstruct it precisely — what signals did I weigh, what did I conclude?

3. **Where did my reasoning fail?**
   - Did I overweight a strength and ignore a fatal flaw?
   - Did I overweight a flaw that reviewers apparently forgave?
   - Did I misjudge the paper type and apply the wrong acceptance bar?
   - Did I mistake polish or ambition for substance?
   - Did I mistake a narrow or preliminary result for an insufficient one?
   - Did I miss what made this paper's contribution genuinely novel or genuinely hollow?

4. **Is this a generalizable lesson?** Ask: would this insight help me on a *different* paper I haven't seen yet? If the lesson is only "this specific paper was accepted," it is not a lesson — it is a label. A real lesson abstracts over paper type, failure mode, or judgment move.

5. **Does my current Meta Review already have a relevant lesson?** If yes, is it the lesson's wording that failed, or did I fail to apply it correctly? The fix is different in each case.

---

## How to update the Meta Review

Based on your diagnosis, do exactly one of:

- **Add** a new lesson — when the failure reveals a distinct judgment pattern not covered by any existing lesson.
- **Revise** an existing lesson — when the failure is a sharper version of something already there (same trigger, refined wording or boundary condition).
- **Merge** two lessons — when you notice they are covering the same failure mode redundantly.
- **No change** — if you conclude the error was a reasonable judgment call and not a systematic blind spot. This is a valid outcome. Not every wrong prediction requires a new lesson.

**Anti-bloat rule:** Do not add a lesson unless it would have prevented this error on a fresh reading. Do not add platitudes ("good papers have strong baselines"). If you cannot state a specific trigger and boundary condition, the lesson is not ready.

---

## Meta Review lesson format

Each lesson must follow this structure:

```
### Lesson N: <short name>

<First-person statement of the judgment move: what I now do differently when I see this pattern>

Applies when: <specific conditions that trigger this lesson>

I guard against: <the exact judgment failure this prevents>

I should NOT apply this when: <boundary condition — when this lesson does not fire>

Confidence: <low | medium | high>
```

---

## Output format

Respond with ONLY the following XML block. Do not add any text before or after it.

```xml
<meta_review_update>
  <diagnosis>
    <prediction_was>ACCEPT or REJECT</prediction_was>
    <correct_label>ACCEPT or REJECT</correct_label>
    <failure_mode>one sentence: what specifically went wrong in my reasoning</failure_mode>
    <is_generalizable>yes or no</is_generalizable>
    <action>add | revise | merge | no_change</action>
  </diagnosis>
  <updated_meta_review>
    [The COMPLETE updated Meta Review text — every lesson in full, including unchanged ones.
     This will be written verbatim to meta_review.txt.
     If no change, reproduce the current Meta Review verbatim.
     If the Meta Review is empty and no lesson is warranted, leave this tag empty.]
  </updated_meta_review>
</meta_review_update>
```

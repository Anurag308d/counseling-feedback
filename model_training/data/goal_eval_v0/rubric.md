# Goal-setting and homework rubric (v0 draft)

Status: first draft, 2026-11-23. Not yet reviewed by a domain expert. Items marked **[GUESS]** are design choices I made without guidance from the brief and should be confirmed with Raj and colleague.

## 1. Where this fits

The FeedbackESConv taxonomy has eight areas (Reflections, Questions, Suggestions, Validation, Self-disclosure, Empathy, Professionalism, Structure). None of them describes how a between-session activity is planned. This rubric adds a ninth area, **Goal Setting**, grounded in behavioral activation (BA): plan specific, values-based activities with the client, schedule them, make them achievable, anticipate barriers, explain why, and review them next time.

## 2. The rubric

A homework or goal-setting response is judged on one gate (G0) and seven items (GS1 to GS7). An item can be met in the helper's last turn **or** in earlier helper turns that the last turn clearly builds on **[GUESS]**. In the eval set every strong response meets all seven in the last turn, so a single turn can be scored on its own.

| ID | Item | Met when the helper... | Typical failure |
|---|---|---|---|
| G0 | Readiness / timing | sets homework only once the problem is understood, acute distress has been heard, there is no unaddressed risk, and the client is open to planning | Jumping to a plan while the client is in crisis, venting, or has just started talking |
| GS1 | Specific and concrete | says what the activity is, when (day / time or trigger), where, and how long or how much | "Try to get out more this week" |
| GS2 | Linked to values | ties the activity to something the client said matters to them (a role, relationship, or value), not a generic "pleasant activity" | Assigning a walk or the gym when the client talked about being a present parent |
| GS3 | Graded and achievable | keeps the first step small enough to succeed on a low-energy day, and breaks bigger goals into steps | "An hour every day plus the weekend" for someone who currently does nothing |
| GS4 | Collaborative | builds on the client's own idea or asks for their input and agreement ("how would that be?"), instead of prescribing | "Here's what you're going to do" / "you need to" |
| GS5 | Anticipates barriers | asks what might get in the way, or plans around an obstacle the client already named | Closing with "you'll do great" |
| GS6 | Explains the rationale | gives a short, plain reason why doing the activity should help (in BA: acting before motivation arrives, activity and mood are linked) | Giving the plan with no reason why it should help |
| GS7 | Plans a follow-up | says when and how the activity will be reviewed (next session, a check-in, what to notice) | No mention of looking back at how it went |

Notes on boundaries **[GUESS]**:

- **Goal Setting vs. Suggestions.** Suggestions covers advice in general ("you should..."). Goal Setting covers planning a specific between-session activity or goal. An imposed homework assignment should be tagged Goal Setting, and Suggestions is an acceptable second tag. A well-formed, collaborative plan should **not** be tagged as a Suggestions weakness. This is the overcorrection the brief hypothesises the current model makes.
- **Goal Setting vs. Structure.** Structure's "[end] Lack of clear, actionable items" overlaps with GS1. I propose: when the helper is (or should be) setting an activity, use Goal Setting. Keep Structure for agenda, focus, and summarising takeaways.
- **Mistimed homework (G0 failures)** gets Goal Setting as a weakness, usually alongside Empathy or Validation.

## 3. Text to add to the annotation prompt

This is written to match the style of `prompts/prompt_gpt_instruction.txt`, so it can go into the "Appendix: Areas for improvement" section and into the LLM-judge prompt.

```
### Goal Setting
- [timing] Setting homework or goals before the seeker feels heard, while they are in acute distress, when they have said they only want to vent, or before the problem is understood
- Making the activity vague instead of specific (what, when, where, how long)
- Choosing an activity that is not connected to what the seeker said matters to them
- Making the first step too big or ambitious rather than small and achievable
- Prescribing the activity instead of choosing it together with the seeker and checking that they agree
- Not asking what might get in the way, or not planning around obstacles the seeker mentioned
- Not explaining why the activity might help
- Not planning when and how to look back at how the activity went
```

As a strength ("goodareas"), Goal Setting means the helper planned a well-timed activity that meets GS1 to GS7.

## 4. Expected labels in the eval set

| Response kind | `perfect` | `badareas` | `goodareas` |
|---|---|---|---|
| strong | true | (none) | must include Goal Setting |
| degraded (any one item broken) | false | must include Goal Setting; `imposed` also accepts Suggestions | |
| negative control, appropriate reply | true | (none) | e.g. Empathy, Validation, Reflections, Questions |
| negative control, mistimed homework | false | must include Goal Setting; Empathy or Validation also expected | |

Gold `feedback` and `alternative` text are **not** written yet. Each degraded row has a one-sentence `flaw` describing what was broken, which the LLM judge can use for the diagnosis metric (task 3b). The brief asks for an expert to write or verify gold feedback for a subset; I suggest 30 rows (one strong plus the full degradation set for about four contexts, plus every negative control).

## 5. Open questions for Raj and colleague

1. Is one turn the right unit? Real BA planning is spread over several turns. I packed all seven items into the last helper turn so each row can be scored alone, which makes strong responses long (around 90 to 120 words, much longer than typical ESConv helper turns). The alternative is to let earlier turns carry some items, which is more natural but harder to label.
2. Should G0 (timing) be part of Goal Setting, or a separate area?
3. Are the boundary rules above (Suggestions, Structure) the ones you want?
4. Should the eval use the "Seeker/Helper" labels even for counselor-client BA excerpts? I kept them so the prompts match the model's training format.
5. Who is the domain expert for verification, and how many rows can they look at?

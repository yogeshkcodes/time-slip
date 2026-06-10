# Start here — Time Slip explained simply

No jargon. What this is, how to run it, what the results mean, and how to talk
about it.

---

## 1. The big idea

Everyone knows *that* they get distracted. Almost nobody knows **why** — was it
boredom? stress? tiredness? the phone itself? dreading the task? Screen-time
apps just show totals. Time Slip is built to answer the real question:

> **What causes *your* attention slips, how much does each cause contribute,
> and what single change would cut them the most?**

It does this with a machine-learning model that watches patterns in your day
(what you were doing, how it felt, your phone context, time-of-day, time-on-task)
and produces your personal **slip fingerprint** — a breakdown like:

- Phone pull: **49%**
- Fatigue: 29%
- Hunger: 13%
- Low motivation: 8%
- ...

…so instead of "be more disciplined," you get "*your* problem is the phone and
sleep — fix those two."

## 2. Why you can trust it (the 30-second version)

Claiming to know *causes* is a big claim, so it's graded two ways:

1. **A ground-truth benchmark.** You can't grade cause-finding on everyday data —
   when someone grabs their phone, nobody can verify the true reason. So the
   method is first graded where the right answers *are* knowable: a benchmark
   world built from cognitive science where the true causes are set in advance.
   The model reverse-engineered them at **98% accuracy**. (This is the standard
   way causal methods are validated.)

2. **Real people.** Then the model's story was tested against a published study
   of **274 real adults** pinged ~8×/day for a week. Boredom, tiredness, low
   interest, stress and low mood all pushed real mind-wandering up, exactly as
   the model says — **83% agreement**. Fun wrinkle for your paper: *effort*
   turned out to be protective — putting in effort means you're engaged.

3. **It tries to break itself.** The project ships a "refutation suite" — five
   tests designed to *catch* a model that's secretly cheating (feed it scrambled
   answers, a fake cause, a do-nothing change…). A fooled model fails them; this
   one passes **5/5**. That's the difference between "looks impressive" and
   "actually holds up."

So: the method finds the right answers where answers are checkable, real humans
behave the way it predicts, and it survives deliberate attempts to break it.
That's the trust story — and it's stronger than almost anything in this space.

## 3. How you actually use it

### Watch your real behaviour (zero effort)
```
python track_me.py --minutes 90     # records which app is in front + idle time
python analyze_tracker.py           # -> your real focus spells, slips, rabbit holes
```
100% on your machine. Nothing is uploaded, ever.

### Get your fingerprint (light logging)
```
python -m timeslip.schema           # makes a blank log template
python analyze_me.py my_log.csv     # -> your personal slip report
```
Fill a row every ~15–30 min (1–5 scales: bored? stressed? tired? phone nearby?
did you slip?). After a couple of weeks you get *your* fingerprint.

### Or log inside Obsidian
```
python obsidian_sync.py "C:/path/to/Vault" --init   # creates a TimeSlip folder
python obsidian_sync.py "C:/path/to/Vault"          # writes a report back, with charts
```

### Ask it anything, live
```
python whatif.py --boredom 4 --task deep_work --tot 40
```
→ "Risk of a slip in the next 10 min: **97%** (87–99%). Best lever right now:
make the task more engaging (−37%). Second: phone away + notifications off (−32%)."

### Reproduce the research study
```
python run_all.py        # ~2.5 min -> all validation, figures, reports
```

## 4. The headline findings

- **Notifications are the biggest lever.** Simulated week-long policies (a true
  causal test, same people with vs without): batching/silencing notifications
  cut slips by ~41%; combined with phone-away and slightly more sleep, **time
  lost dropped ~46%**.
- **It's not the ping itself — it's you.** A notifications-only predictor gets
  ROC 0.60; adding internal states (boredom, stress, fatigue, task feelings)
  takes it to **0.76–0.77**. Most of the signal is internal.
- **Two kinds of "cause," two different answers.** What makes you *prone* to
  slipping: boredom and depleted self-control. What actually *tips you over* in
  the moment: the phone pull and dreading the task. Most advice confuses these.
- **The fingerprint is personal.** A notification-heavy manager: 56% phone pull.
  A disciplined morning person: task aversiveness. Same "distraction," opposite fixes.

## 5. How to tell people about it

**One sentence:**
> "I built a system that figures out what actually causes a specific person's
> attention slips — validated against ground truth and corroborated on data from
> 274 real people."

**30 seconds:**
> "Screen-time apps tell you *that* you got distracted; mine tells you *why* and
> *what to change*. It gives each person a 'slip fingerprint' — how much of their
> distraction comes from the phone pull vs boredom vs fatigue vs stress. Causal
> claims need grading, so I graded it where ground truth is knowable (98%
> recovery) and then checked the story against a published study of 274 adults —
> 83% agreement. It also ships as working tools: an on-device behaviour tracker,
> an Obsidian integration, and a live risk coach."

**The punchline:**
> "Attention is becoming the scarce currency. This measures its exchange rate —
> person by person — and tells you the cheapest place to buy yours back."

---

### In one line
**Grade the method where truth is knowable → confirm the story in real humans →
then point it at your own life.**

(Technical details: [README.md](README.md) · theory & citations: [docs/THEORY.md](docs/THEORY.md))

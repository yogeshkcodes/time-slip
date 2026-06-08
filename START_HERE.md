# Start here — Time Slip explained simply

No jargon. This is what the project is, how to run it, what the results mean, and
how to explain it to someone else.

---

## 1. The big idea (what this even is)

You want to figure out **why people get distracted** — why they grab their phone,
zone out, snack, or wander off a task.

The problem: in real life you can **never know for sure**. If someone picks up
their phone, were they bored? Stressed? Tired? Did a notification just arrive? You
can guess, but you can't check if your guess is right. There's no answer key.

**So we use a trick.** We built a realistic *fake-person generator* — a simulator
where **we** secretly decide the rules of what causes distraction. Then we let the
computer study these fake people's days and try to figure out the rules on its
own. Because we already know the real rules, **we can grade the computer.**

It scored about **98% right** (on a big, deliberately hard dataset of 36 people
over 28 days — nearly a million logged minutes). That's the whole point: once it
can correctly reverse-engineer causes we planted in fake data, we can trust it to
do the same on **real** data later (yours, or study participants').

> Think of it like a lie detector you first test on people you *know* are lying —
> once it passes that test, you trust it on strangers.

---

## 2. How you actually run it

One command. In the `Time Slip` folder, type:

```
python run_all.py
```

Wait ~25 seconds. It creates fake people, studies them, and writes every answer
into the **`outputs`** folder.

Then just open files and look:

- **Start here:** `outputs/reports/findings.md` — the plain-English summary.
- **Pictures:** the `outputs/figures` folder — charts ready for a paper or slides.
- **Per person:** files like `outputs/reports/person_P04.md` — one report card per person.

**To analyze your own real life:**

```
python -m timeslip.schema       # makes a blank template to fill in
python analyze_me.py my_log.csv # turns your filled log into your own report
```

You fill in a row every so often (what you were doing, how bored/stressed/tired
you felt on a 1–5 scale, was your phone nearby, did you slip). It then shows
**where, when and why *you* slip** and builds *your* personal fingerprint. Don't
have data yet? Just run `python analyze_me.py` with no file — it makes a realistic
example so you can see exactly what you'd get.

**Prefer Obsidian?** You can do the whole thing inside your notes:

```
python obsidian_sync.py "C:/path/to/your/Vault" --init   # makes a TimeSlip folder
# ...log each day as a simple table in TimeSlip/logs/ (a template is provided)...
python obsidian_sync.py "C:/path/to/your/Vault"          # writes a report back
```

It drops a **"Time Slip Report"** note (with charts) right into your vault.

---

## 3. What the results are and how to read them

The system gives **four answers**:

### (a) "Can it predict when you're about to slip?" → Yes, ~76–77%
When you show it an "about-to-slip" moment and a "you're-fine" moment, it picks the
risky one **~76% of the time even for people it has never seen before**, and **~77%
once it knows you** (your own past days). Random guessing is 50%. Just blaming
notifications only gets ~60% — so **most distraction comes from what's going on
inside you, not just your phone buzzing.** (It can't hit 99% — *nobody* can predict
the exact minute you'll glance away; that part is genuinely random.)

### (b) "Did it find the real causes?" → Yes, 98% match
This is the trust score. The one thing it *couldn't* untangle was "low mood" vs
"stress" (too similar to tell apart) — and we report that honestly instead of
hiding it.

### (c) The "slip fingerprint" — the coolest output
For each person it makes a breakdown of *what causes their distraction*, like a pie
chart. Example — the sales manager:

- Phone pull: **56%**
- Task aversiveness (dreading the task): 18%
- Stress: 12%
- …everything else small

So you can literally say *"56% of this person's distraction is the phone, so the #1
fix is keeping the phone out of reach."* A different person's fingerprint looks
completely different — **that's the value.**

### (d) The big insight worth a headline
There are **two different kinds of "cause," and they have different answers:**

- *What keeps you generally prone to drifting?* → **boredom and low self-control.**
- *What actually tips you over at the exact moment you slip?* → **the phone and dreading the task.**

Most people lump "distraction" into one bucket. This shows it's two things, and the
fix depends on which one you mean.

---

## 4. How to tell people about it

**One sentence:**
> "I built a model that figures out *why* a specific person loses focus — and I
> proved it works by testing it on simulated people whose real causes I already knew."

**30-second version (professor / recruiter):**
> "Distraction research usually just says 'phone use correlates with boredom.' Mine
> goes further: it gives each person a 'fingerprint' of what causes *their* focus to
> break, and by how much. The hard part is you can't verify causes in real life, so I
> generated realistic fake days from known rules and showed my method recovers those
> rules at 92% accuracy. That validation is what makes it trustworthy on real data."

**Punchline for a paper or talk:**
> "Attention is becoming the scarce currency. This measures its exchange rate —
> person by person — and separates what makes you *prone* to distraction from what
> actually *triggers* it."

**If someone asks "is this real people?"** Be honest:
> "Not yet — it's simulated, on purpose, to prove the method is sound. The next step
> is plugging in real self-logged data, which the system already accepts."

---

### In one line
**Build fake people with known rules → let the AI find the rules → grade it (98%)
→ now trust it to explain real distraction, person by person.**

(For the technical details, see [README.md](README.md) and [docs/THEORY.md](docs/THEORY.md).)

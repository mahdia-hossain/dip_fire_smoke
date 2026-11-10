# Fire / Smoke / Normal Detection — The Whole Project Explained Simply

*For someone who has never heard of image processing or AI before.*
*Updated: 9 November 2026*

---

## 1. The project in one minute (say this to anyone)

> "I am building a computer program that looks at a photo and tells you whether it shows **fire**,
> **smoke**, or **nothing dangerous (normal)**. It should work indoors and outdoors — in buildings,
> factories and streets — so that in the future a robot or a camera could raise an alarm by itself.
>
> The hard part is not real fire. The hard part is things that **look like** fire or smoke: a sunset,
> a street lamp, car headlights, fog. A bad program raises an alarm for every sunset.
>
> My method has two parts. First, classic **image processing** cleans the photo, finds the
> fire-coloured and smoke-coloured areas, and measures simple things like colour, brightness and texture.
> Second, a small and fast **AI model** (a transformer) makes the final decision — and I give it the
> measurements from the first part as **hints**. Then I test whether those hints make it fooled less often
> by look-alikes."

---

## 2. A few words you need first

| Word | Simple meaning | Everyday example |
| --- | --- | --- |
| **Pixel** | One tiny coloured dot. A photo is a grid of pixels. | A 256 × 256 photo has 65,536 dots. |
| **Digital Image Processing (DIP)** | Changing or measuring a photo with fixed maths rules. | The "brightness" and "contrast" sliders in a phone's photo editor. |
| **Dataset** | A big collection of photos, each with a correct answer (label). | A box of flashcards with the answer written on the back. |
| **Label** | The correct answer for a photo: fire, smoke or normal. | The answer on the back of a flashcard. |
| **AI model** | A program that learns patterns from many labelled examples. | A student who studies thousands of flashcards. |
| **Training** | Showing the model labelled photos so it learns. | The student practising with the flashcards. |
| **Testing** | Checking the model on photos it has **never seen**. | The final exam with new questions. |
| **Transformer** | A type of AI model that cuts the photo into small squares and lets every square "look at" every other square. | Understanding a sentence by seeing how all the words relate to each other. |
| **False alarm** | The model says "fire" when there is no fire. | A smoke alarm going off because of toast. |
| **Look-alike** | Something that is not fire/smoke but looks like it. | Sunset, lamp, headlights, fog. |

---

## 3. The pipeline — step by step

A **pipeline** is a series of steps, like a factory line: a photo goes in at one end and the answer comes out
at the other end. Our pipeline has 7 steps (Step 0 to Step 6).

```
Photo → [0 Dataset] → [1 Clean the photo] → [2 Find fire/smoke areas] → [3 Measure it with numbers]
      → [4 AI decides, with hints] → [5 Test it] → [6 Final tool: answer + highlighted areas]
```

---

### Step 0 — Preparing the dataset  ✅ DONE

**What it does:** collects thousands of photos, gives each one the correct label, removes copies, and divides
them into "study" photos and "exam" photos.

**Why:** an AI can only learn from examples, and an exam is only fair if the questions were not in the
study material.

**What we did:**
- We used **three public datasets**:
  - **D-Fire** — about 21,500 photos, mostly outdoor.
  - **FireAndSmoke** — about 27,300 photos: indoor, outdoor, buildings, factories, cars, plus look-alike photos
    (lamps, sunsets, headlights).
  - **DeepQuestAI** — about 3,000 general photos, kept **completely separate** as the final exam.
- **One label rule for all photos:** fire = any fire in the photo; smoke = only smoke; normal = no fire or smoke.
- **Removed copies.** Many photos are frames from the same video, so two photos can be almost identical.

**Example:** a video of a burning truck gives 50 frames that look almost the same. If frame 10 is in the
study set and frame 11 is in the exam, the exam is too easy — the model has basically seen the answer. So our
program compares every photo with every other photo (using a "fingerprint" of each image) and removes the
study copy when the exam has a near-copy.

**What we found (useful for the report):** **5,872** D-Fire study photos were near-copies of D-Fire exam
photos. The original dataset split was "leaking" answers. We removed them.

**Final numbers:**

| Set | Fire | Smoke | Normal | What it is for |
| --- | --- | --- | --- | --- |
| Train | 12,837 | 6,239 | 4,971 | The model studies these |
| Validation | 4,805 | 1,922 | 598 | Mock tests while studying |
| Test-in | 2,858 | 1,930 | 2,022 | Exam from the same sources |
| DeepQuestAI | 940 | 993 | 998 | Final exam from a different source |
| Look-alike test | — | — | 451 | Trick questions (see below) |

---

### Step 0b — The look-alike test set  ✅ DONE

**What it does:** builds a special exam made only of trick photos — things that look like fire or smoke but
are not.

**Why:** this is our strongest research point. Most papers report one overall score. We want to know
*exactly which look-alikes* fool the model.

**What we did:** you labelled each look-alike photo by type in a small browser app (press 1–5):

| Type | Photos | Example |
| --- | --- | --- |
| Sunset / sunrise | 88 | Orange sky at sunset |
| Lamp / street light | 52 | A street lamp at night |
| Vehicle / emergency light | 185 | Car headlights, fire-truck lights |
| Reflection / glow / screen | 51 | Lights reflected on a wet road, a neon sign |
| Cloud / fog / haze | 75 | Grey clouds, fog over a city |

You also found **19 photos that actually showed real fire or smoke** but were labelled "look-alike" in the
original dataset — mistakes in the dataset. We removed them. The set is now **frozen** (locked), so it cannot
be changed after we start testing models.

---

### Step 1 — Preprocessing (cleaning the photo)  🔜 code ready, run next

**What it does:** makes every photo the same size and a bit cleaner and clearer.

**Why:** photos come in many sizes and qualities. Cleaning them makes the later steps fairer and more reliable.

It has three small steps:

1. **Resize to 256 × 256 pixels.**
   *Example:* a 640 × 640 phone photo and a 416 × 416 camera photo both become 256 × 256.
   *Analogy:* cutting all papers to A4 before filing them.

2. **Median filter (3 × 3) — removes tiny noise dots.**
   For every pixel, it looks at the 9 pixels around it (a 3 × 3 square), sorts their values and takes the
   middle one.
   *Example:* one random white dot in a dark area gets replaced by the normal dark colour around it.
   *Analogy:* in a group survey, ignoring one strange answer and keeping the typical one.
   *Why median and not blur:* the median is the best of the three at removing salt-and-pepper dots (dead pixels, sensor specks). On edges it is about the same as a Gaussian blur (E02: edge preservation 0.79 vs 0.80); the bilateral filter keeps edges best (0.85) but is slower and weaker on dots.

3. **CLAHE — improves contrast, region by region.**
   CLAHE (Contrast Limited Adaptive Histogram Equalization) makes dark areas clearer by spreading out the
   brightness levels, separately in each small region of the photo. We apply it to **brightness only**, so the
   colours of fire do not change.
   *Example:* in a dark factory photo, the dim smoke near the ceiling becomes easier to see.
   *Analogy:* turning up the brightness only in the dark corners of a photo, not the whole photo.

**Measuring how much the photo changed — PSNR and SSIM (sir's request):**
we compare **the photo we put in** with **the photo we get after preprocessing**.
- **PSNR** (Peak Signal-to-Noise Ratio), in decibels: higher = less change. Above ~40 dB the change is
  almost invisible; around 25 dB it is clearly visible.
- **SSIM** (Structural Similarity), from 0 to 1: 1 = the shapes and structure are exactly the same.
- *Example:* after the median filter, PSNR might be ~45 dB and SSIM ~0.99 (tiny change). After CLAHE, PSNR
  might be ~25 dB and SSIM ~0.85 (clear contrast change, but the same scene).
- **Important:** PSNR/SSIM tell us **how much** a step changes the photo, not whether the result is "better".

**Extra test (to justify our choices, which the rubric rewards):** we add known noise to a clean photo on
purpose, then check which filter removes it best (median, Gaussian blur or bilateral). We also check which
contrast method changes the colours least (CLAHE on brightness vs other methods).

---

### Step 2 — Segmentation (finding where the fire and smoke are)  🔜 next

**What it does:** marks *where* in the photo the fire-coloured and smoke-coloured areas are. The result is a
black-and-white picture called a **mask**: white = suspicious area, black = everything else.

**How:**
- **Fire mask (YCbCr colour rule).** YCbCr is another way to describe colour: Y = brightness, Cb = blueness,
  Cr = redness. Fire pixels are **bright and more red than blue**, so we mark the pixels that follow this rule.
  *Analogy:* a guard told "stop anyone wearing bright orange".
- **Smoke mask (HSV + dark channel).** HSV describes colour as hue (which colour), saturation (how strong) and
  value (how bright). Smoke is **greyish** (low saturation) and **hazy**. The **dark channel** measures haze: in a
  clear photo, almost every small area has a very dark pixel; smoke makes everything milky, so that dark value
  goes up. *Analogy:* looking through a dirty window — even black things look grey.
- **Otsu's method** picks the best cut-off value automatically (hazy vs not hazy).
  *Analogy:* a machine that sorts apples into big and small by finding the best size limit itself.
- **Morphology** cleans the mask: **opening** erases tiny white specks; **closing** fills small holes.
  *Analogy:* cleaning a drawing — erase stray dots, then fill gaps in the shapes.

**Example:** in a photo of a burning car, the fire mask lights up the flames, and the smoke mask lights up the
grey cloud above the car.

**Note for the report:** these masks show *candidate* areas; they are used for pictures and for measurements.
They are not graded as an object detector.

---

### Step 3 — Feature extraction (describing the photo with numbers)  🔜 next

**What it does:** turns each photo into about **16 numbers** that describe it the way an image-processing
expert would.

| Group | What it measures | Example of why it helps |
| --- | --- | --- |
| **Colour** | How red/orange, how grey, share of fire-coloured and smoke-coloured pixels | Fire is orange-red; smoke is grey |
| **Brightness** | How bright, how much contrast | Flames are very bright |
| **Texture** (LBP, GLCM) | Rough or smooth, patterned or plain | Smoke is soft and swirly; a lamp glow is smooth |
| **Haze** (dark channel) | How milky/foggy the photo looks | Smoke makes the photo hazy |
| **Edges** (Sobel) | How many sharp edges there are | Flames have sharp, flickering edges; fog has almost none |

**Example:** a sunset might score *high orange* but *low texture* and *few sharp edges*; real fire scores
*high orange*, *rough texture* and *many sharp edges*. These differences are exactly the hints that should help
the AI not be fooled.

*Analogy:* describing a person with height, weight, age and hair colour instead of showing their photo.

---

### Step 4 — The models (the deciders)  🔜 later

**What it does:** decides **fire, smoke or normal**. We build five versions and compare them fairly
(same photos, same settings):

| Model | What it looks at | Simple description |
| --- | --- | --- |
| **A. SVM / Random Forest** | Only the 16 numbers | Classic method. Random Forest asks many yes/no questions ("Is it very orange? Is it hazy?") and takes a vote |
| **B. MobileNetV3** | The cleaned photo | A small CNN (another type of AI) that reads the photo with a sliding magnifying glass |
| **C. MobileViT-XS** | The cleaned photo | A small, fast transformer — our baseline |
| **D. MobileViT-XS + numbers at the end** | Photo + 16 numbers | The transformer looks at the photo; the numbers are added just before the final answer |
| **E. MobileViT-XS + DIP hints inside ★** | Photo + 16 numbers | **Our main model.** The 16 numbers become extra "hint tokens" inside the transformer, so it can use them at every thinking step |

*Analogy for E vs D:* in D, the student finishes the exam and then glances at an expert's notes before
handing it in. In E, the expert sits next to the student for the whole exam and whispers hints.

**Why compare D and E?** To be honest: if the simple way (D) works just as well, we report that.

---

### Step 5 — Evaluation (the exams)  🔜 later

**What it does:** gives every model three exams and scores them.

- **Exam 1 — Test-in:** new photos from the same sources as training.
- **Exam 2 — DeepQuestAI:** photos from a completely different source (real-world check).
- **Exam 3 — Look-alikes:** the 451 trick photos.

**Scores (in simple words):**
- **Accuracy:** % of photos answered correctly.
- **Macro-F1:** a fair score that treats fire, smoke and normal equally, even though there are more fire photos.
- **Confusion matrix:** a table showing what got mixed up with what (e.g. how many smoke photos were called normal).
- **Fire recall:** of all real fires, how many were caught — the most important number for safety.
- **False alarms per look-alike type:** e.g. "called 12% of sunsets fire".
- **Speed (FPS):** how many photos per second — important for a robot.

---

### Step 6 — The final tool  🔜 later

**What it does:** `predict.py` — you give it **any new photo**, and it returns:
- the answer (fire / smoke / normal) and how sure the model is,
- the photo with the fire and smoke areas highlighted,
- a small result file.

*This is the "finished product" — what a robot would run on every camera picture in the future.*

---

## 4. Why this is research, not just a project (the research gaps)

A **research gap** is a problem that published papers say is still not solved. We use four papers:

| Paper | What it showed | What is missing (the gap) | What we do |
| --- | --- | --- | --- |
| **D-ViT** (2025) | A transformer can sort fire / smoke / normal | It uses pixels only — no image-processing knowledge | We give the transformer DIP hints |
| **STKformer** (2025) | One image-processing clue (dark channel) helps a transformer | Smoke only, one clue | Many clues, all three classes |
| **FireAndSmoke** (2024) | Look-alikes (lamps, sunsets, headlights) cause false alarms | Models are rarely tested on them per type | Our look-alike test, per type |
| **Scientific Reports** (2026) | Small transformers are fast but can miss fires in hard lighting | — | We check whether DIP hints help them catch more fires |

**Honest wording:** we say "we **test whether** DIP hints help", never "we prove".

---

## 5. How far we have come

| Step | Status | Result |
| --- | --- | --- |
| Research gaps, papers, plan | ✅ Done | 4 key papers, idea fixed |
| Step 0 — Dataset | ✅ Done | 3 datasets combined; copies removed; answer leakage found and fixed |
| Step 0b — Look-alike test | ✅ Done | 451 trick photos in 5 types, frozen; 19 dataset mistakes found |
| Project setup (training guide) | ✅ Done | One command (`run.py`) for every step; every run saved with full details |
| Step 1 — Preprocessing + PSNR/SSIM | 🔜 Code ready, next run | — |
| Step 2 — Segmentation | 🔜 Next | — |
| Step 3 — Features | 🔜 Next | — |
| Step 4 — Models A–E | 🔜 12–15 Nov | — |
| Step 5 — Evaluation | 🔜 16 Nov | — |
| Step 6 — Final tool | 🔜 17 Nov | — |
| Report + presentation | 🔜 18–20 Nov | — |

---

## 6. Questions people might ask you (and simple answers)

**"Why not just use an AI on the photo?"**
Because AI models learn from pixels only and can be fooled by things that look like fire. Image-processing
measurements (colour, texture, haze, edges) are clues that stay true everywhere. We test whether combining both
works better.

**"Why a small model?"**
A robot or camera has little computing power. A small model like MobileViT-XS can run fast enough on small devices.

**"Why so much effort on removing copies?"**
Many photos are frames from the same video. If a near-copy is in both study and exam, the score looks better
than it really is. We found and removed 5,872 such leaks in one dataset.

**"What is PSNR/SSIM for?"**
To show how much each cleaning step changes the photo: we compare the photo we put in with the photo we get out.

**"What is new in your work?"**
Putting many image-processing clues *inside* a small transformer for fire / smoke / normal, and testing it
honestly on look-alikes, on a different dataset, and for speed.

**"Is it only forest fires?"**
No. The data covers indoor fires, buildings, factories, cars and streets, plus wildfires.

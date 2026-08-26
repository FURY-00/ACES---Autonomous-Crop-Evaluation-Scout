# Studying the tuning at home, with no Pi and no camera

Two new tools. Both run on a laptop with Python + OpenCV.

    pip install opencv-python numpy

---

## The one idea worth understanding

Every decision the detector makes is a threshold on a single number:

    d = (G - R) / (R + G + B)

Multiply all three channels by the same factor -- which is what shading,
tilt and exposure do -- and d does not change. That is why it survives
lighting changes when hue does not.

    healthy green     d ~ +0.15 .. +0.30
    chlorotic yellow  d ~  0.00 .. +0.06
    necrotic brown    d ~ -0.06 .. +0.02

Three thresholds are applied to it, and they do different jobs:

| parameter | question it answers |
|---|---|
| `min_d_ref` | is this a plant at all? |
| `d_healthy_foliage` | does this leaf have any healthy tissue to compare against? |
| `d_abs_max` | how un-green must a pixel be to count as a lesion? |

`bg_k` and `exg_thresh` come earlier and decide the leaf outline. If those
are wrong, the three above are being computed on the wrong pixels and no
amount of tuning them will help.

---

## Tool 1 — `explain.py`, understand ONE image

    python3 tools/explain.py leaf.jpg

Prints a text histogram of d inside the leaf, marks where each threshold
falls, states which gate accepted or rejected the frame, and suggests
values.

**Reading the histogram is the skill.** Look at the shape:

    two humps    healthy tissue and lesion, cleanly separable.
                 Put d_abs_max in the valley between them.

    one hump     the leaf is uniformly one thing. If that hump sits low,
                 the whole leaf is diseased -- that is the whole-leaf path.

    smeared      no clean separation. Usually glare, blur, or a leaf mask
                 that leaked into the background. Fix that, not the threshold.

If the histogram has a clean valley and the detector still misses, the
problem is a gate, and section 4 of the output names which one.

---

## Tool 2 — `offline_tune.py`, find values that work on MANY images

Sliders in a field optimise for the one leaf in front of you. That is
exactly how you get settings that fail on the next leaf. This searches a
grid and scores every combination against a labelled folder.

    dataset/
      diseased/     images that DO have abnormal tissue
      healthy/      clean leaves
      notleaf/      hands, soil, sky, shirts, buildings

    python3 tools/offline_tune.py dataset/ --quick     # ~1 min
    python3 tools/offline_tune.py dataset/             # thorough
    python3 tools/offline_tune.py dataset/ --write     # save the winner

It prints the top combinations with precision and recall, then names every
image it got wrong -- so you can go and look at those specific files.

**The `notleaf/` folder is the important one.** It is what stops the search
picking settings that flag everything. Without negatives, "flag everything"
scores perfect recall and the search will happily choose it.

Precision is weighted above recall on purpose. A false positive costs a
wasted stop, a wasted capture and a wrong pin on a farmer's map. A miss
costs one leaf out of hundreds.

Then copy the result over:

    scp config/detector_tuned.json pi4@<pi-ip>:~/acesss/aces/config/

---

## Where to get images without the Pi

**Best: the Pi already saved them.** Every detection wrote a full-resolution
original to `data/<date>/raw/`. Next time you have access:

    scp -r pi4@<pi-ip>:~/acesss/aces/data ./aces_data

Those are real Pi Camera frames in real conditions -- far better than
anything you can shoot at home.

**Second best: your phone.** Photograph leaves against different backgrounds
and in different light. Not the same sensor, and phones apply their own
saturation and HDR, so absolute d values will differ from the Pi's. Fine for
learning the shape of the problem, not for fixing final numbers.

**For negatives, no camera needed at all.** Hands, soil, sky, clothing,
buildings -- you already have photos of all of these. Every one you add
makes the search harder to fool.

---

## A study session that will actually teach you something

1. Put 5 diseased, 5 healthy and 5 not-leaf images in a dataset folder.
2. Run `explain.py` on one diseased image. Read the histogram. Find the
   valley between the humps.
3. Run it on a healthy image. Notice there is no second hump.
4. Run it on a hand. Notice d is near zero everywhere -- that is what
   `min_d_ref` exists to catch.
5. Run `offline_tune.py --quick`. Compare the winner to what you set by
   hand in the field.
6. Look at whichever images it got wrong. **That is where the learning is.**
   Usually one bad file -- blurred, mislabelled, badly lit -- is dragging
   the whole search.

---

## What is worth improving, in order

1. **More images, especially negatives.** Every parameter is being fitted to
   your data. Fifteen images is enough to start; fifty is enough to trust.
2. **Images from more than one lighting condition.** Your indoor tuning
   failed outdoors because it had only ever seen one light.
3. **Only then, the parameters.** They are the last thing to optimise, not
   the first. A clean leaf mask on varied data beats a perfectly tuned
   threshold on one photo.

The failure you had in the field was never really a threshold problem. It
was tuning on a sample of one.

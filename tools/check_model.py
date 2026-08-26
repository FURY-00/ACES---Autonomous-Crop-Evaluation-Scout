"""
Verify your trained model BEFORE the demo.

The failure this catches is nasty and silent: if labels.txt is not in the
same order as the model's output indices, the robot names the wrong disease
with total confidence. Nothing crashes. You find out when a teacher asks
why a healthy leaf is being called late blight.

Usage
-----
    python3 tools/check_model.py                       # load + inspect only
    python3 tools/check_model.py path/to/leaf.jpg      # + classify one image
    python3 tools/check_model.py testset/diseased/     # + classify a folder

If you have images whose class you KNOW, run the folder mode and check the
predictions match. That is the only real test of label order.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S          # noqa: E402


def main():
    print("=" * 66)
    print("ACES model check")
    print("=" * 66)

    # ---- 1. runtime -------------------------------------------------
    Interp = None
    for mod, name in (("tflite_runtime.interpreter", "tflite-runtime"),
                      ("tensorflow.lite", "tensorflow"),
                      ("ai_edge_litert.interpreter", "ai-edge-litert")):
        try:
            Interp = __import__(mod, fromlist=["Interpreter"]).Interpreter
            print(f"  runtime: {name}")
            break
        except ImportError:
            continue
    if Interp is None:
        print("  NO TFLITE RUNTIME. Try, in this order:")
        print("     pip install tflite-runtime --break-system-packages")
        print("     pip install ai-edge-litert --break-system-packages")
        print("     sudo apt install python3-tflite-runtime")
        print("\n  The demo still runs without it -- detections are logged as")
        print("  'abnormal_leaf' with no disease name.")
        return

    # ---- 2. files ----------------------------------------------------
    print(f"\n  model:  {S.TFLITE_MODEL}")
    if not os.path.exists(S.TFLITE_MODEL):
        print("  NOT FOUND. Copy your .tflite there, e.g.:")
        print(f"     cp ~/plant_disease.tflite {S.TFLITE_MODEL}")
        return
    print(f"          {os.path.getsize(S.TFLITE_MODEL)/1e6:.1f} MB")

    labels = []
    if os.path.exists(S.LABELS_FILE):
        with open(S.LABELS_FILE) as fh:
            labels = [l.strip() for l in fh if l.strip()]
        print(f"  labels: {S.LABELS_FILE}  ({len(labels)} classes)")
    else:
        print(f"  labels: {S.LABELS_FILE}  NOT FOUND")

    # ---- 3. load -----------------------------------------------------
    it = Interp(model_path=S.TFLITE_MODEL)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    n_out = int(out["shape"][-1])
    side = int(inp["shape"][1])

    print(f"\n  input  {tuple(inp['shape'])}  {inp['dtype'].__name__}")
    print(f"  output {tuple(out['shape'])}  {n_out} classes")

    if side != S.CLASSIFIER_INPUT:
        print(f"  ! settings.CLASSIFIER_INPUT is {S.CLASSIFIER_INPUT} but the "
              f"model wants {side}. Fix it in config/settings.py.")
    if labels and len(labels) != n_out:
        print(f"  ! MISMATCH: {len(labels)} labels vs {n_out} model outputs.")
        print("    The robot WILL report wrong disease names. Fix labels.txt")
        print("    before the demo -- one class name per line, in the exact")
        print("    order your training script used (usually alphabetical by")
        print("    folder name; check your train script's class_indices).")
    elif labels:
        print("  labels count matches output count")

    # ---- 4. speed ----------------------------------------------------
    dummy = (np.random.rand(1, side, side, 3) * 255)
    dummy = dummy.astype(np.uint8) if inp["dtype"] == np.uint8 \
        else dummy.astype(np.float32) / 255.0
    it.set_tensor(inp["index"], dummy)
    it.invoke()                                   # warm up
    t0 = time.time()
    for _ in range(5):
        it.set_tensor(inp["index"], dummy)
        it.invoke()
    ms = (time.time() - t0) / 5 * 1000
    print(f"\n  inference: {ms:.0f} ms per image")
    if ms > 800:
        print("  ! slow. Fine for the demo (the robot is stopped anyway),")
        print("    but too slow to run per frame.")

    # ---- 5. real images ----------------------------------------------
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("\n  Pass an image or folder to test real predictions:")
        print("     python3 tools/check_model.py testset/diseased/")
        return

    from perception import detector as D
    from perception.classifier import DiseaseClassifier
    clf = DiseaseClassifier()
    if not clf.ok:
        return

    paths = ([os.path.join(target, f) for f in sorted(os.listdir(target))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
             if os.path.isdir(target) else [target])
    print(f"\n{'image':30s} {'prediction':28s} {'conf':>6s}  runner-up")
    print("-" * 78)
    for p in paths[:15]:
        img = cv2.imread(p)
        if img is None:
            continue
        small = cv2.resize(img, (800, 450))
        r = D.detect(small)
        sx, sy = img.shape[1] / 800.0, img.shape[0] / 450.0
        blobs = [{"x": int(b["x"] * sx), "y": int(b["y"] * sy),
                  "w": int(b["w"] * sx), "h": int(b["h"] * sy)}
                 for b in r.blobs]
        name, conf, top = clf.predict(img, blobs)
        second = f"{top[1][0]} {top[1][1]:.0%}" if len(top) > 1 else ""
        mark = " " if conf >= S.CONF_UNCERTAIN else "?"
        print(f"{os.path.basename(p)[:29]:30s} {name[:27]:28s} "
              f"{conf:5.0%}{mark} {second}")

    print("\n  '?' = below CONF_UNCERTAIN, would go to data/uncertain/ and NOT")
    print("        be sent to Telegram.")
    print("\n  Do these names match what you know the images to be? If they are")
    print("  consistently wrong but consistently the SAME wrong class, your")
    print("  labels.txt order is off.")


if __name__ == "__main__":
    main()

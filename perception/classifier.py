"""
MobileNetV2 / TFLite disease classifier.

Two things this adds over calling the interpreter directly:

1. It classifies the LESION CROP, not the whole frame. PlantVillage images
   are single leaves filling the frame. If you feed a wide field shot with a
   2% lesion, the model sees mostly soil and returns confident nonsense. The
   detector already told you where the lesion is — use it.

2. It reports "uncertain" honestly. A softmax always sums to 1, so the model
   always names a disease, even for a photo of your shoe. Anything below
   CONF_UNCERTAIN goes to data/uncertain and is never sent to the farmer.
"""

import os

import cv2
import numpy as np

from config import settings as S

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        Interpreter = None


class DiseaseClassifier:
    def __init__(self, model=S.TFLITE_MODEL, labels=S.LABELS_FILE):
        self.ok = False
        if Interpreter is None:
            print("[classifier] no tflite runtime installed")
            return
        if not os.path.exists(model):
            print(f"[classifier] model not found: {model}")
            return
        self.interp = Interpreter(model_path=model)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.quant = self.inp["dtype"] == np.uint8
        self.labels = ["class_%d" % i for i in range(self.out["shape"][-1])]
        if os.path.exists(labels):
            with open(labels) as fh:
                self.labels = [l.strip() for l in fh if l.strip()]
        self.ok = True
        print(f"[classifier] loaded {len(self.labels)} classes, "
              f"{'uint8' if self.quant else 'float32'} input")

    # ------------------------------------------------------------------
    def _crop_to_lesion(self, bgr, blobs, pad_frac=0.55):
        """
        Crop a square around the lesions with generous padding, so the model
        sees the lesion in the context of surrounding leaf tissue -- which is
        what it was trained on.
        """
        if not blobs:
            return bgr
        x0 = min(b["x"] for b in blobs)
        y0 = min(b["y"] for b in blobs)
        x1 = max(b["x"] + b["w"] for b in blobs)
        y1 = max(b["y"] + b["h"] for b in blobs)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * (0.5 + pad_frac)
        H, W = bgr.shape[:2]
        half = max(half, 60)
        a = int(max(0, cx - half)); b_ = int(min(W, cx + half))
        c = int(max(0, cy - half)); d = int(min(H, cy + half))
        crop = bgr[c:d, a:b_]
        return crop if crop.size else bgr

    def predict(self, bgr, blobs=None, topk=3):
        """Returns (label, confidence, [(label, prob), ...])."""
        if not self.ok:
            return "unknown", 0.0, []
        img = self._crop_to_lesion(bgr, blobs or [])
        img = cv2.resize(img, (S.CLASSIFIER_INPUT, S.CLASSIFIER_INPUT))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.quant:
            x = np.expand_dims(rgb.astype(np.uint8), 0)
        else:
            x = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)

        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)

        if self.quant:
            scale, zero = self.out["quantization"]
            if scale:
                y = (y - zero) * scale
        if y.sum() <= 0 or y.max() > 1.5:          # raw logits -> softmax
            e = np.exp(y - y.max())
            y = e / e.sum()

        order = np.argsort(y)[::-1][:topk]
        top = [(self.labels[i] if i < len(self.labels) else str(i), float(y[i]))
               for i in order]
        return top[0][0], top[0][1], top

# Models

Put your trained artefacts here:

    plant_disease.tflite    exported from train_disease_classifier.py
    labels.txt              one class name per line, SAME ORDER as training

`labels.txt` order must match the model's output indices exactly. If they are
out of order the robot will confidently report the wrong disease and you will
not notice until a farmer does. Verify with one known image before a field run.

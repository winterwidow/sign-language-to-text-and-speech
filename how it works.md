collect_images.py:

1. - creates a ./data directory with subfolders named 0, 1, ..., 32—each representing a unique sign.
2. - For each class:

- It shows a live video feed and waits for you to press "Q" when you're ready.
- Then it captures 100 frames and saves them as .jpg images in the corresponding folder.

This gives you a clean, labeled image dataset that you’ll later process with MediaPipe to extract hand landmarks.

---

create_dataset.py:

- Loops through each image in your ./data folder.
- Uses MediaPipe to detect hand landmarks (21 key points per hand).
- Extracts the x and y coordinates of each landmark.
- Normalizes them by subtracting the minimum x and y values (to make the data position-independent).
- Stores the processed landmark data along with its class label.
  Output
  A data.pickle file containing:
- data: a list of normalized landmark vectors (your features)
- labels: the corresponding class labels (e.g., "A", "B", "Hello")
  This transforms raw images into structured numerical data that a machine learning model can understand.

---

train_classifier.py:
It trains a Random Forest classifier to recognize hand signs based on the landmark data you extracted earlier using MediaPipe.

- Loads the data.pickle file containing normalized hand landmark features and labels.
- Flattens each landmark vector into a single feature array (so it's compatible with scikit-learn).
- Splits the data into training and testing sets (80/20), ensuring balanced class distribution.
- Trains a RandomForestClassifier on the training data.
- Evaluates the model on the test set and prints the accuracy.
- Saves the trained model to model.p for later use in real-time prediction.

Output

- A printed accuracy score.
- A serialized model file (model.p) ready for deployment.

---

inference_classifier.py:

It uses your webcam to detect hand signs live, extracts landmark features using MediaPipe, and feeds them into your trained Random Forest model to predict the sign. The result is displayed directly on the video feed.

- Loads the trained model from model.p.
- Captures frames from your webcam.
- Uses MediaPipe to detect hand landmarks.
- Normalizes the landmark coordinates.
- Predicts the sign using your model.
- Draws a bounding box and overlays the predicted label on the video stream.

Output

- A live video window showing your hand.
- Predicted sign label (e.g., "A", "Hello") displayed above the hand.
- Bounding box drawn around the detected hand.

---

# ✋ GestureFlow – Sign Language Recognition & Game

GestureFlow is a **real-time sign language recognition system** that converts gestures into **text and speech**, with an additional **multiplayer game mode** where two players race to sign letters.  

It uses **Flask-SocketIO**, **MediaPipe Hands**, **OpenCV**, and a trained ML model to process webcam input, predict gestures, and provide **text & audio output**.

---

## 🚀 Features
- 🔤 **Sign-to-Text** – Recognizes hand gestures (A–Z) and converts them into text in real-time.  
- 🔊 **Sign-to-Speech** – Converts recognized text into spoken audio.  
- 🎮 **Game Mode – Lexicon**  
  - Player vs. Player (split-screen).  
  - Race to sign the prompted letter.  
  - Timer & scoring system with live updates.  
- ⚡ **Fast Processing** – Uses WebSockets for smooth predictions.  
- ✨ **Modern UI** – Clean TailwindCSS design, glowing effects, and smooth split layout.  

---

## 🛠 Requirements

- Python **3.9** (recommended for MediaPipe compatibility)
- Virtual environment (`venv`)
- Dependencies listed in `requirements.txt`:
  ```txt
  flask
  flask-socketio
  mediapipe
  opencv-python
  numpy
  pyttsx3   # for text-to-speech

# EyeSnap

EyeSnap is a gaze-aware auto-photo application that uses your webcam to track your eye movement.  
When you look away from the camera or move out of the frame, EyeSnap automatically captures photos.  
This project uses [MediaPipe](https://mediapipe.dev/) for facial landmark detection and PyQt6 for the GUI.

## Features

- Real-time eye tracking with MediaPipe Face Mesh  
- Detects whether you are looking at the camera or away  
- Automatically takes snapshots when you look away or leave the frame  
- Desktop notifications on photo capture  
- Easy-to-use graphical user interface with live webcam feed and status display  


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/akash-jithu/EyeSnap.git
   cd EyeSnap

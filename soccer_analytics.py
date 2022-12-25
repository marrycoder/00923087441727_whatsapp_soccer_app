import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
  # Handle missing values
  df = df.fillna(method='ffill')

  # Encode categorical variables
  encoder = LabelEncoder()
  df['player'] = encoder.fit_transform(df['player'])
  df['position'] = encoder.fit_transform(df['position'])
  df['action'] = encoder.fit_transform(df['action'])

  # Scale numerical variables
  scaler = StandardScaler()
  df[['frame', 'player', 'position']] = scaler.fit_transform(df[['frame', 'player', 'position']])

  return df


def extract_player_positions(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect player positions using object detection
    player_positions = {}
    player_detector = cv2.CascadeClassifier('player_detector.xml')
    players = player_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in players:
        player_positions[(x + w//2, y + h//2)] = (x, y, w, h)

    return player_positions


def extract_player_actions(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract player actions using object tracking
    player_actions = {}
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray, gray, p0, None, **lk_params)
        for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        player_actions[(c, d)] = (a - c, b - d)

    return player_actions


# Load the soccer video
video = cv2.VideoCapture('soccer.mp4')

# Initialize an empty DataFrame to store the data
df = pd.DataFrame(columns=['frame', 'player', 'position', 'action'])

# Extract the relevant information from the video
while video.isOpened():
  # Read the next frame
  success, frame = video.read()
  if not success:
    break

  # Extract the player positions and actions from the frame
  player_positions = extract_player_positions(frame)
  player_actions = extract_player_actions(frame)

  # Store the data in the DataFrame
  for i, player in enumerate(player_positions):
    df = df.append({'frame': frame, 'player': player, 'position': player_positions[player], 'action': player_actions[player]}, ignore_index=True)

# Close the video
video.release()

# Preprocess the data
df = preprocess_data(df)

# Save the DataFrame to a CSV file
df.to_csv('soccer_video_data.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas

# Load the soccer video data
df = pd.read_csv('soccer_video_data.csv')

# Split the data into training and testing sets
X = df[['frame', 'player', 'position']]
y = df['action']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a random forest classifier on the data
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Generate a classification report
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)

# Create a PDF report
c = canvas.Canvas('soccer_analytics_report.pdf')

# Add the classification report to the PDF
textobject = c.beginText()
textobject.setTextOrigin(50, 750)
textobject.setFont('Helvetica', 12)
textobject.textLines(report)
c.drawText(textobject)

# Save the PDF
c.showPage()
c.save()



import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- SETUP STORAGE ---
DB_PATH = "students.csv"
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

if not os.path.exists(DB_PATH):
    pd.DataFrame(columns=['Name', 'RollID', 'Gender']).to_csv(DB_PATH, index=False)

# --- AI SETUP (Ultra-Lightweight) ---
# This uses almost 0 RAM and never crashes Streamlit
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    st.error("System booting up AI components...")

def train_ai():
    faces, ids = [], []
    df = pd.read_csv(DB_PATH)
    for _, row in df.iterrows():
        path = f"{FACES_DIR}/{row['RollID']}.jpg"
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                ids.append(int(row['RollID']))
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.save("brain.yml")

st.set_page_config(page_title="Smart Student Portal", layout="centered")
st.title("🏫 AI Attendance Portal")

tab1, tab2, tab3 = st.tabs(["1. Register", "2. Mark Attendance", "3. Report"])

# --- TAB 1: REGISTRATION ---
with tab1:
    st.header("Register New Student")
    st.write("⚠️ *Roll Number MUST be numbers only (e.g., 47)*")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        roll = st.text_input("Roll Number (Numbers only)")
        gender = st.radio("Gender", ["Boy", "Girl"])
    with col2:
        img_buffer = st.camera_input("Snap your registration photo")

    if st.button("REGISTER STUDENT", type="primary"):
        if not name or not roll or not roll.isdigit():
            st.error("⚠️ Please enter a Name and a NUMERIC Roll ID.")
        elif img_buffer is None:
            st.error("⚠️ Please snap a photo first!")
        else:
            bytes_data = img_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(detected) == 0:
                st.error("❌ No face detected! Please look directly at the camera with good lighting.")
            else:
                (x, y, w, h) = detected[0]
                # Crop and resize to standard size for better training
                face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                
                # Save
                cv2.imwrite(f"{FACES_DIR}/{roll}.jpg", face_img)
                df = pd.read_csv(DB_PATH)
                df = df[df['RollID'].astype(str) != str(roll)]
                new_row = pd.DataFrame([{'Name': name, 'RollID': roll, 'Gender': gender}])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(DB_PATH, index=False)
                
                train_ai()
                st.success(f"✅ SUCCESS: {name} is now registered!")

    st.write("### Current Database")
    if os.path.exists(DB_PATH):
        st.dataframe(pd.read_csv(DB_PATH), use_container_width=True)

# --- TAB 2: ATTENDANCE ---
with tab2:
    st.header("Mark Attendance")
    att_buffer = st.camera_input("Verify your face for attendance")

    if st.button("RECOGNIZE & MARK PRESENT", type="primary"):
        if not os.path.exists("brain.yml"):
            st.error("❌ No AI Brain found. Please register at least one student first!")
        elif att_buffer is None:
            st.error("⚠️ Snap a photo first!")
        else:
            recognizer.read("brain.yml")
            bytes_data = att_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(detected) == 0:
                st.error("❌ No face detected. Make sure your face is clearly visible.")
            else:
                recognized = False
                for (x, y, w, h) in detected:
                    roll_id, confidence = recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
                    
                    # LBPH Confidence: Lower is better. Usually < 85 is good.
                    if confidence < 85: 
                        df = pd.read_csv(DB_PATH)
                        match = df[df['RollID'].astype(str) == str(roll_id)]
                        if not match.empty:
                            student = match.iloc[0]
                            log_file = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
                            pd.DataFrame([{'Name': student['Name'], 'ID': student['RollID'], 'Gender': student['Gender'], 'Time': datetime.now().strftime('%H:%M:%S')}]).to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
                            st.success(f"📝 MARKED PRESENT: {student['Name']}")
                            recognized = True
                            break
                if not recognized:
                    st.error("❌ Face not recognized. Are you registered? (If yes, try better lighting).")

# --- TAB 3: REPORT ---
with tab3:
    st.header("Dropout Analysis Report")
    if st.button("Generate Today's Report"):
        if not os.path.exists(DB_PATH):
            st.warning("No students in the database.")
        else:
            m = pd.read_csv(DB_PATH)
            l = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
            if not os.path.exists(l):
                st.warning("No attendance marked today yet.")
            else:
                att = pd.read_csv(l).drop_duplicates(subset=['ID'])
                st.markdown("### 📊 Daily Dropout Rates")
                for g in ["Boy", "Girl"]:
                    tot = len(m[m['Gender'] == g])
                    pre = len(att[att['Gender'] == g])
                    rate = ((tot - pre) / tot * 100) if tot > 0 else 0
                    st.markdown(f"- **{g}s:** {pre}/{tot} present (**{rate:.1f}% Absent/Dropout**)")

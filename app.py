import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
import numpy as np

# --- 1. SETUP STORAGE ---
DB_PATH = "students.csv"
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Create the Excel file if it doesn't exist
if not os.path.exists(DB_PATH):
    pd.DataFrame(columns=['Name', 'RollID', 'Gender']).to_csv(DB_PATH, index=False)

def get_list():
    if os.path.exists(DB_PATH):
        return pd.read_csv(DB_PATH)
    return pd.DataFrame(columns=['Name', 'RollID', 'Gender'])

st.set_page_config(page_title="Smart Student Portal", layout="centered")
st.title("🏫 Guaranteed Student Portal")

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["1. Register Student", "2. Mark Attendance", "3. Dropout Analysis"])

# --- TAB 1: REGISTRATION ---
with tab1:
    st.header("Register a New Student")
    
    col1, col2 = st.columns(2)
    with col1:
        n_in = st.text_input("Full Name")
        r_in = st.text_input("Roll Number")
        g_in = st.radio("Gender", ["Boy", "Girl"])
    
    with col2:
        # Streamlit's built-in camera is incredibly stable
        img_file_buffer = st.camera_input("Snap photo first!")

    if st.button("REGISTER", type="primary"):
        if not n_in or not r_in:
            st.error("⚠️ ERROR: Type your Name and Roll ID!")
        elif img_file_buffer is None:
             st.error("⚠️ ERROR: Click the camera icon to snap a photo!")
        else:
             try:
                # Read the image buffer into an OpenCV format
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                save_path = os.path.join(FACES_DIR, f"{r_in}.jpg")
                cv2.imwrite(save_path, cv2_img)
                
                df = pd.read_csv(DB_PATH)
                df = df[df['RollID'].astype(str) != str(r_in)] # Overwrite if exists
                new_row = pd.DataFrame([{'Name': n_in, 'RollID': r_in, 'Gender': g_in}])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(DB_PATH, index=False)
                
                # Clear AI memory
                cache = os.path.join(FACES_DIR, "representations_vgg_face.pkl")
                if os.path.exists(cache):
                    os.remove(cache)
                    
                st.success(f"✅ SUCCESS: {n_in} is registered!")
             except Exception as e:
                 st.error(f"❌ CRASH: {str(e)}")
                 
    st.write("### Current Database")
    st.dataframe(get_list(), use_container_width=True)

# --- TAB 2: ATTENDANCE ---
with tab2:
    st.header("Mark Attendance")
    att_img_buffer = st.camera_input("Take a photo to mark attendance")
    
    if st.button("RECOGNIZE FACE", type="primary"):
        if att_img_buffer is None:
            st.error("❌ Snapshot required.")
        else:
            try:
                # Load DeepFace only when needed
                from deepface import DeepFace
                
                bytes_data = att_img_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                temp_path = "temp_check.jpg"
                cv2.imwrite(temp_path, cv2_img)
                
                results = DeepFace.find(img_path=temp_path, db_path=FACES_DIR, enforce_detection=False, model_name="VGG-Face")
                
                if len(results) > 0 and not results[0].empty:
                    match_path = results[0].iloc[0]['identity']
                    roll = os.path.basename(match_path).split('.')[0]
                    
                    df = pd.read_csv(DB_PATH)
                    student = df[df['RollID'].astype(str) == str(roll)].iloc[0]
                    
                    log_name = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
                    pd.DataFrame([{'Name': student['Name'], 'ID': student['RollID'], 'Gender': student['Gender'], 'Time': datetime.now().strftime('%H:%M:%S')}]).to_csv(log_name, mode='a', header=not os.path.exists(log_name), index=False)
                    
                    st.success(f"📝 PRESENT: {student['Name']} (ID: {student['RollID']})")
                else:
                    st.error("❌ NOT RECOGNIZED: Face not in database.")
            except Exception as e:
                 st.warning("⚠️ AI Warming Up... Please wait 5 seconds and click again.")

# --- TAB 3: REPORT ---
with tab3:
    st.header("Dropout Analysis")
    if st.button("Run Analysis"):
        if not os.path.exists(DB_PATH): 
            st.warning("No students registered.")
        else:
            m = pd.read_csv(DB_PATH)
            l = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
            if not os.path.exists(l): 
                st.warning("No attendance logged today.")
            else:
                att = pd.read_csv(l).drop_duplicates(subset=['ID'])
                st.markdown("### 📊 Dropout Analysis")
                for g in ["Boy", "Girl"]:
                    tot = len(m[m['Gender'] == g])
                    pre = len(att[att['Gender'] == g])
                    rate = ((tot - pre) / tot * 100) if tot > 0 else 0
                    st.markdown(f"- **{g}s:** {pre}/{tot} present ({rate:.1f}% Dropout)")

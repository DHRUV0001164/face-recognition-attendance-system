import cv2
import numpy as np
import face_recognition
import streamlit as st
import pickle
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Face Attendance", layout="wide")

ENC_FILE = "encodings.pkl"
ATT_FILE = "attendance.csv"
TOLERANCE = 0.5

# ---------------- Helpers ----------------
def load_encodings():
    if os.path.exists(ENC_FILE):
        with open(ENC_FILE, "rb") as f:
            data = pickle.load(f)
        return {name: [np.array(e) for e in encs] for name, encs in data.items()}
    return {}

def save_encodings(encs):
    serial = {name: [e.tolist() for e in lst] for name, lst in encs.items()}
    with open(ENC_FILE, "wb") as f:
        pickle.dump(serial, f)

def save_attendance(name, course, year):
    now = datetime.now()
    df = pd.DataFrame([[name, course, year, now.strftime("%H:%M:%S"), now.strftime("%d-%m-%Y")]],
                      columns=["Name","Course","Year","Time","Date"])
    if os.path.exists(ATT_FILE):
        old = pd.read_csv(ATT_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(ATT_FILE, index=False)

def match_face(known, enc):
    valid_vecs, valid_names = [], []
    for name, lst in known.items():
        for e in lst:
            if isinstance(e, np.ndarray) and e.shape == (128,):
                valid_vecs.append(e)
                valid_names.append(name)
    if not valid_vecs:
        return None
    dists = face_recognition.face_distance(valid_vecs, enc)
    idx = np.argmin(dists)
    return valid_names[idx] if dists[idx] <= TOLERANCE else None

# ---------------- Session State ----------------
if "encodings" not in st.session_state:
    st.session_state.encodings = load_encodings()
if "student_info" not in st.session_state:
    st.session_state.student_info = {}  # {name: {"course":..,"year":..}}

st.title("Face Recognition Attendance System")

col1, col2 = st.columns(2)

# ---------------- CAMERA ----------------
with col1:
    st.subheader("Live Camera Feed")
    start_cam = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    if start_cam:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, face_locs)
            display_names = []

            for enc in encs:
                name = match_face(st.session_state.encodings, enc)
                if name:
                    info = st.session_state.student_info.get(name, {"course":"Unknown","year":"Unknown"})
                    save_attendance(name, info["course"], info["year"])
                    display_names.append(name)
                else:
                    display_names.append("Unknown")

            # Annotate faces
            for (top, right, bottom, left), nm in zip(face_locs, display_names):
                color = (0,255,0) if nm!="Unknown" else (0,0,255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, nm, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            FRAME_WINDOW.image(frame, channels="BGR")
        else:
            st.error("Cannot access camera!")

# ---------------- REGISTER FACE ----------------
with col2:
    st.subheader("Register New Face")
    name = st.text_input("Enter Name")
    course = st.selectbox("Select Course", ["B.Tech CSE","B.Tech IT","B.Sc CS","BCA","MCA"])
    year = st.selectbox("Select Year", ["1st Year","2nd Year","3rd Year","4th Year"])
    if st.button("Capture & Register"):
        if name.strip()=="":
            st.warning("Enter a valid name!")
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                st.error("Camera not found!")
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                encs = face_recognition.face_encodings(rgb, locs)
                if not encs:
                    st.error("No face detected!")
                else:
                    enc = encs[0]
                    if name not in st.session_state.encodings:
                        st.session_state.encodings[name] = []
                    st.session_state.encodings[name].append(enc)
                    st.session_state.student_info[name] = {"course":course,"year":year}
                    save_encodings(st.session_state.encodings)
                    st.success(f"{name} ({course}, {year}) registered successfully!")

# ---------------- REGISTERED PEOPLE ----------------
st.markdown("---")
st.subheader("Registered People")
if st.session_state.encodings:
    for n in st.session_state.encodings.keys():
        info = st.session_state.student_info.get(n, {"course":"Unknown","year":"Unknown"})
        st.write(f"• {n} ({info['course']}, {info['year']})")
else:
    st.write("_No faces registered yet._")

# ---------------- ATTENDANCE ----------------
st.markdown("---")
st.subheader("Attendance saved in attendance.csv")
if os.path.exists(ATT_FILE):
    df = pd.read_csv(ATT_FILE)
    st.dataframe(df)
else:
    st.write("_No attendance recorded yet._")

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
import time
from ultralytics import YOLO
from retinaface import RetinaFace
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from difflib import get_close_matches
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
STATIC_FOLDER = './static'
RESULT_FOLDER = "./results"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# Load models
yolo_v8 = YOLO('yolov8s-oiv7.pt')
orb = cv2.ORB_create(nfeatures=1000)
flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2), dict(checks=50))
embedder = FaceNet()

@app.route('/')
def home():
    return render_template('index.html')


# ---------- COMMON HELPERS ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def milliseconds_to_hms(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60
    return f"{int(hours):02}:{int(minutes % 60):02}:{int(seconds % 60):02}"

# # ---------- APP1: Object Detection (YOLO) ----------
@app.route('/object_detection')
def object_detection_index():
    return render_template('index1.html')

@app.route('/object_detection/upload', methods=['POST'])
def object_detection_upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index1.html', error_message="No selected file")
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        object_name = request.form['object_name']
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        detected_frames = find_object_in_video(video_path, object_name, confidence_threshold)
        if detected_frames:
            return render_template('results1.html', frames=detected_frames)
        return render_template('index1.html', error_message="No objects found matching your criteria.")
    return render_template('index1.html', error_message="Invalid file format.")


def find_object_in_video(video_path, object_name, confidence_threshold=0.5, match_threshold=20):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    last_frame_descriptors = None
    detected_frames_info = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = yolo_v8.predict(frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                label = yolo_v8.names[class_id]

                if label.lower() == object_name.lower() and confidence >= confidence_threshold:
                    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
                    if descriptors is None or len(descriptors) < 2:
                        continue

                    if last_frame_descriptors is not None:
                        matches = flann.knnMatch(last_frame_descriptors, descriptors, k=2)
                        good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.7 * n.distance]
                        if len(good_matches) >= match_threshold:
                            continue
                    
                    last_frame_descriptors = descriptors
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label_text = f"{label}: {confidence*100:.1f}%"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 2)

                    raw_timestamp = frame_count / fps
                    hours = int(raw_timestamp // 3600)
                    minutes = int((raw_timestamp % 3600) // 60)
                    seconds = int(raw_timestamp % 60)
                    formatted_timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"

                    result_frame = f"frame_{frame_count:04d}.jpg"
                    cv2.imwrite(os.path.join(RESULT_FOLDER, result_frame), frame)

                    detected_frames_info.append({
                    'frame': frame_count,
                    'timestamp': formatted_timestamp,
                    'label': label,
                    'frame_file': result_frame,
                    })
                    break  
    video.release()
    return detected_frames_info


@app.route('/object_detection/output_frames/<filename>')
def serve_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# screenshot detection
@app.route("/screenshot_matching", methods=["GET", "POST"])
def screenshot_matching():
    if request.method == "POST":
        video = request.files["video"]
        image = request.files["image"]
        if not video or not image:
            return render_template("index2.html", error="Please upload both files.")
        
        video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
        image_path = os.path.join(UPLOAD_FOLDER, secure_filename(image.filename))
        match_img_path = os.path.join(STATIC_FOLDER, "match.jpg")

        video.save(video_path)
        image.save(image_path)

        timestamp, error = find_image_in_video(video_path, image_path, match_img_path, match_threshold=90)

        if error:
            return render_template("results2.html", match_found=False, error=error)
        
        return render_template("results2.html", match_found=True, timestamp=timestamp, match_img="match.jpg")
    
    return render_template("index2.html")



def find_image_in_video(video_path, template_path, output_path, match_threshold=90):
    orb = cv2.ORB_create(nfeatures=1500)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    template = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_kp, template_desc = orb.detectAndCompute(template_gray, None)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = orb.detectAndCompute(frame_gray, None)

        if frame_desc is None:
            continue

        matches = flann.knnMatch(template_desc, frame_desc, k=2)
        good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.6 * n.distance]

        if len(good_matches) >= match_threshold:
            # Homography check
            src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                inliers = np.sum(mask)
                if inliers >= match_threshold:
                    timestamp = format_time(frame_count / fps)
                    match_img = cv2.drawMatches(template, template_kp, frame, frame_kp, good_matches[:20], None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imwrite(output_path, match_img)
                    video.release()
                    return timestamp, None

    video.release()
    return None, "No match found in the video."





# ---------- APP3: Face Matching (FaceNet + RetinaFace) ----------


@app.route('/person_matching')
def person_matching_index():
    return render_template('index3.html')

@app.route('/person_matching/upload', methods=['POST'])
def person_matching_upload():
    reference_image = request.files['reference_image']
    video_files = request.files.getlist('videos')
    threshold = float(request.form['threshold'])
    reference_image_path = os.path.join(UPLOAD_FOLDER, secure_filename(reference_image.filename))
    reference_image.save(reference_image_path)
    match_results = []
    for video in video_files:
        video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
        video.save(video_path)
        match_data = detect_face_in_video(video_path, reference_image_path, threshold)
        match_results.extend(match_data)
    return render_template('results3.html', match_results=match_results)

def detect_face_in_video(video_path, reference_image_path, threshold):
    cap = cv2.VideoCapture(video_path)
    ref_embedding, _ = load_reference_embedding(reference_image_path)
    match_data = []
    # SKIP INTERVAL YHA HAI----------------------------------
    frame_skip_interval = 70 
    frame_count = 0
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue
        if last_frame is not None and are_frames_similar(last_frame, frame):
            continue
        small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        faces = RetinaFace.detect_faces(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        for key in faces:
            x1, y1, x2, y2 = [coord * 2 for coord in faces[key]['facial_area']]
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            embedding = get_face_embedding(face_crop)
            similarity = cosine_similarity([ref_embedding], [embedding])[0][0]
            if similarity >= threshold:
                timestamp = format_time(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

                # Draw green bounding box around detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Overlay similarity percentage
                similarity_text = f"{int(similarity * 100)}%"
                cv2.putText(frame, similarity_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                img_encoded = base64.b64encode(buffer).decode('utf-8')
                match_data.append({
                    'video': os.path.basename(video_path),
                    'timestamp': timestamp,
                    'image': img_encoded
                })
                last_frame = frame
    cap.release()
    return match_data

def load_reference_embedding(path):
    faces = RetinaFace.detect_faces(path)
    key = list(faces.keys())[0]
    x1, y1, x2, y2 = faces[key]['facial_area']
    ref_img = cv2.imread(path)[y1:y2, x1:x2]
    return embedder.embeddings([cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)])[0], (x1, y1, x2, y2)

def get_face_embedding(face):
    return embedder.embeddings([cv2.cvtColor(face, cv2.COLOR_BGR2RGB)])[0]

def are_frames_similar(f1, f2):
    kp1, des1 = orb.detectAndCompute(f1, None)
    kp2, des2 = orb.detectAndCompute(f2, None)
    if des1 is None or des2 is None:
        return False
    matches = flann.knnMatch(np.uint8(des1), np.uint8(des2), k=2)
    good = [m[0] for m in matches if len(m) > 1 and m[0].distance < 0.7 * m[1].distance]
    return len(good) > 10


# ---------- APP4: Speech Transcription and Keyword Search ----------


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_audio_from_video(video_file, audio_file):
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

def split_audio(audio_file, chunk_length_ms):
    audio = AudioSegment.from_wav(audio_file)
    chunks = []
    overlap = 3000  
    start = 0
    while start < len(audio):
        end = start + chunk_length_ms
        chunks.append(audio[start:end])
        start += chunk_length_ms - overlap
    return chunks

def transcribe_audio_chunk(chunk, recognizer, language='en-US'):
    with chunk.export(format="wav") as source:
        audio = sr.AudioFile(source)
        with audio as audio_source:
            audio_data = recognizer.record(audio_source)
            try:
                text = recognizer.recognize_google(audio_data, language=language)
                return text
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                return ""


def search_word_in_transcript(transcript, search_words, chunk_start_time, chunk_length_ms, fuzzy_match, cutoff):
    timestamps = {}
    word_times = {}
    words = transcript.lower().split()
    search_words = [word.lower() for word in search_words]

    for i, word in enumerate(words):
        if word in search_words:
            word_time = chunk_start_time + (i / len(words)) * chunk_length_ms
            word_time_hms = milliseconds_to_hms(word_time)
            if word in timestamps:
                timestamps[word].append(word_time_hms)
            else:
                timestamps[word] = [word_time_hms]

            if word not in word_times:
                word_times[word] = []
            word_times[word].append(word_time)
        elif fuzzy_match:
            similar_words = get_close_matches(word, search_words, n=1, cutoff=cutoff)
            if similar_words:
                closest_word = similar_words[0]
                word_time = chunk_start_time + (i / len(words)) * chunk_length_ms
                word_time_hms = milliseconds_to_hms(word_time)
                if closest_word in timestamps:
                    timestamps[closest_word].append(word_time_hms)
                else:
                    timestamps[closest_word] = [word_time_hms]

                if closest_word not in word_times:
                    word_times[closest_word] = []
                word_times[closest_word].append(word_time)

    concatenated_words = []
    concatenated_times = []
    sorted_words = sorted(word_times.keys(), key=lambda w: min(word_times[w]) if word_times[w] else float('inf'))

    i = 0
    while i < len(sorted_words) - 1:
        current_word = sorted_words[i]
        current_times = word_times[current_word]
        j = i + 1
        concatenated = current_word
        min_time = min(current_times)

        while j < len(sorted_words):
            next_word = sorted_words[j]
            next_times = word_times[next_word]

            if abs(min(next_times) - min(current_times)) <= 1000: 
                concatenated += " " + next_word
                j += 1
                current_times = next_times
                min_time = min(min_time, min(current_times))
            else:
                break

        if concatenated != current_word:
            concatenated_words.append(concatenated)
            concatenated_times.append(milliseconds_to_hms(min_time))

        i = j

    return timestamps, concatenated_words, concatenated_times

def transliterate_search_words(words):
    transliterated_words = []
    for word in words:
        try:
            hindi_script = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            transliterated_words.append(hindi_script)
        except Exception as e:
            transliterated_words.append(word)
    return transliterated_words

# Home page
@app.route('/search_in_video')
def search_in_video_index():
    return render_template('index4.html')

@app.route('/search_in_video/process', methods=['POST'])
def search_in_video_process():
    if 'video' not in request.files:
        return render_template('error4.html', message="No video file uploaded")
    
    video = request.files['video']
    if video.filename == '':
        return render_template('error4.html', message="No video selected for uploading")

    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        language = request.form.get('language', 'en')
        search_string = request.form.get('search_string', '')
        fuzzy_match = request.form.get('fuzzy_match', 'no') == 'yes'
        cutoff = float(request.form.get('cutoff', 0.8))

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
        extract_audio_from_video(video_path, audio_path)

        recognizer = sr.Recognizer()
        chunks = split_audio(audio_path, 15000)

        search_words = search_string.split()
        if language == 'hi':
            search_words = transliterate_search_words(search_words)

        timestamps = {}
        chunk_transcriptions = []
        for i, chunk in enumerate(chunks):
            chunk_start_time = i * (15000 - 3000)
            text = transcribe_audio_chunk(chunk, recognizer, language='hi-IN' if language == 'hi' else 'en-US')
            
            chunk_transcriptions.append({'chunk': i + 1, 'transcription': text})
            word_timestamps, concatenated_words, concatenated_times = search_word_in_transcript(
                text, search_words, chunk_start_time, 15000, fuzzy_match, cutoff
            )

            for word, times in word_timestamps.items():
                if word in timestamps:
                    timestamps[word].extend(times)
                else:
                    timestamps[word] = times

            for concatenated, time in zip(concatenated_words, concatenated_times):
                if concatenated not in timestamps:
                    timestamps[concatenated] = [time]

        # Render results on the webpage
        return render_template(
            'results4.html',
            transcriptions=chunk_transcriptions,
            timestamps=timestamps,
            search_words=search_words
        )
    else:
        return render_template('error4.html', message="Allowed video formats are mp4, avi, mov, mkv")
    

# VIDEO TO PDF MAKER -------------------------------------------------------------------
@app.route("/video_to_pdf", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            return "No file uploaded", 400

        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        pdf_name = f"{os.path.splitext(filename)[0]}.pdf"
        pdf_path = os.path.join(RESULT_FOLDER, pdf_name)
        result = video_to_pdf(video_path, pdf_path, interval_seconds=10)

        os.remove(video_path)

        if result:
            return redirect(url_for("result", file=pdf_name))
        else:
            return "No frames extracted", 500

    return render_template("index5.html")

@app.route("/video_to_pdf/result")
def result():
    file = request.args.get("file")
    if not file:
        return "No file specified", 400
    return render_template("results5.html", file=file)

@app.route("/video_to_pdf/download/<file>")
def download(file):
    path = os.path.join(RESULT_FOLDER, file)
    return send_file(path, as_attachment=True)
def video_to_pdf(video_path, output_pdf, interval_seconds=10):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return None

    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image_path = f"{UPLOAD_FOLDER}/frame_{frame_count}.jpg"
            cv2.imwrite(image_path, frame)
            saved_frames.append(image_path)

        frame_count += 1

    video.release()

    if not saved_frames:
        return None

    images = [Image.open(f).convert("RGB") for f in saved_frames]
    images[0].save(output_pdf, save_all=True, append_images=images[1:])

    for f in saved_frames:
        os.remove(f)

    return output_pdf

# ---------- RUN APP ----------
if __name__ == '__main__':
    import sys
    port = 5000
    for arg in sys.argv:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
    app.run(debug=True, port=port)

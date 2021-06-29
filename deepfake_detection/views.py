import os
import cv2
import dlib
import shutil
import datetime
import random
import math
import secrets
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings
from keras.models import load_model
from natsort import natsorted
from numpy.lib.function_base import place
from tensorflow.python.keras.backend import placeholder

if not os.path.exists(settings.TEMP_ROOT):
    os.makedirs(settings.TEMP_ROOT)

# load models
inception_path = os.path.join(settings.MODEL_ROOT, 'InceptionV3_Face2Face.h5')
inception = load_model(inception_path)

xception_path = os.path.join(settings.MODEL_ROOT, 'Xception_Face2Face.h5')
xception = load_model(xception_path)

densenet_path = os.path.join(settings.MODEL_ROOT, 'Densenet_Face2Face.h5')
densenet = load_model(densenet_path)


# TRUNCATE DECIMALS 
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


# CLEAR TEMPORARY FOLDER
def clear_temp(token):

    dir = os.path.join(settings.TEMP_ROOT, token)
    if os.path.exists(dir):
        shutil.rmtree(dir)                              # delete user data when useless
        old_dirs = os.listdir(settings.TEMP_ROOT)
        for old_dir in old_dirs:                        # delete old folders that were not deleted for some reason
            path = os.path.join(settings.TEMP_ROOT, old_dir)
            if os.path.exists(path) and datetime.datetime.now().timestamp() - os.path.getctime(path) >= 600:  
                shutil.rmtree(path)


# CHOOSE 6 RANDOM VIDEOS FROM VIDEO DIRECTORY + RELATIVE PLACEHOLDERS
def choose_random_videos():

    manipulated = []
    original = []

    videos = []
    placeholders = []

    manipulated = os.listdir(os.path.join(settings.VIDEO_ROOT, 'manipulated'))
    original = os.listdir(os.path.join(settings.VIDEO_ROOT, 'original'))

    random.shuffle(manipulated)
    random.shuffle(original)

    videos = manipulated[:3] + original[:3]
    random.shuffle(videos)

    i = 0
    for video in videos:
        name = os.path.splitext(video)[0]
        placeholders.append(os.path.join('placeholders', str(name) + '.png'))
        if '_' in name:
            videos[i] = os.path.join( 'video', 'manipulated', video)
        else: 
            videos[i] = os.path.join('video', 'original', video)
        i += 1

    return videos, placeholders


# SAVE VIDEO IN TEMP DIR AND RETURNS ITS PATH AND THE DIR PATH
def get_paths(video, token):

    video_name = os.path.basename(video.name)
    video_name = os.path.splitext(video_name)[0]
    work_dir = os.path.join(settings.TEMP_ROOT, token)

    fs = FileSystemStorage(location=work_dir, base_url=work_dir)
    file = fs.save(video_name + '.mp4', video)                      # save in temp folder
    file = fs.url(file)                                             # get file name
    video_path = os.path.join(work_dir, file)

    return video_path, work_dir


# GET BOUNDING BOX FROM EXTRACTED FACE
def get_boundingbox(face, width, height, scale=1.3):

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    size_bb = int(max(x2 - x1, y2 - y1)*scale)   # size of bounding box

    center_x = (x1 + x2)/2
    center_y = (y1 + y2)/2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


# WRITE FRAME IN THE TARGET DIRECTORY
def write_frame(frame, dir, resize_width, index):

    frame = cv2.resize(frame, (resize_width, 480), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dir, str(index)+'.png'), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# DRAW BOX AROUND THE FACE AND ADD LABEL WITH PREDICTION
def bound_frame(frame, prediction, x, y, size):

    if prediction <= 0.25:
        bounded_frame = cv2.rectangle(frame, (x, y), (x+size, y+size), (0,155,0), 2)
        cv2.putText(bounded_frame, 'Fake: ' + str(round((prediction)*100, 2)) + '%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,155,0), 2)
    elif prediction >= 0.75:
        bounded_frame = cv2.rectangle(frame, (x, y), (x+size, y+size), (0,0,255), 2)
        cv2.putText(bounded_frame, 'Fake: ' + str(round((prediction)*100, 2)) + '%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    else:
        bounded_frame = cv2.rectangle(frame, (x, y), (x+size, y+size), (43,183,252), 2)
        cv2.putText(bounded_frame, 'Fake: ' + str(round((prediction)*100, 2)) + '%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (43,183,252), 2)

    return bounded_frame


# EXTRACTS ALL FRAMES FROM THE VIDEO AND PREDICTS
def extract_and_predict(video, token):

    video_path, work_dir = get_paths(video, token)

    frames_dir = os.path.join(work_dir, 'frames')       # create directory for frames
    if not os.path.exists(frames_dir):     
        os.makedirs(frames_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps, 0))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_lenght = n_frames / fps
    if video_lenght <= 20:
        step = fps / 2
    else:
        step = fps

    face_detector = dlib.get_frontal_face_detector()  # set face detector model

    success, frame = cap.read()
    i = 0   # count frames written
    pred = 0    # count predictions

    height, width, channels = frame.shape
    ratio = width / height
    resize_height = 480
    resize_width = round(ratio * resize_height)

    inception_predictions = []
    xception_predictions = []
    densenet_predictions = []
    predictions_indexes = []

    while success:   # open the video
        if i % int(step) == 0:
            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 0)    # detect faces
            if len(faces):    # select first largest face
                face = faces[0]
                x, y, size = get_boundingbox(face, width, height)   # get bounding box
                face = frame[y:y+size, x:x+size]   

                face_ix = face / 255
                face_ix = cv2.resize(face_ix, (299, 299), interpolation=cv2.INTER_AREA)
                face_ix = face_ix.reshape(1, 299, 299, channels)
                inception_predictions.append(1-inception.predict(face_ix)[0][0])
                xception_predictions.append(1-xception.predict(face_ix)[0][0])
                face_d = face / 255
                face_d = cv2.resize(face_d, (224, 224), interpolation=cv2.INTER_AREA)
                face_d = face_d.reshape(1, 224, 224, channels)
                densenet_predictions.append(1-densenet.predict(face_d)[0][0])
                predictions_indexes.append(i)

                for j in range(i, i+int(step)):
                    if not success: 
                        break
                    bound_frame(frame, xception_predictions[pred], x, y, size)
                    write_frame(frame, frames_dir, resize_width, j)
                    success, frame = cap.read()
                pred += 1 
                i += int(step) - 1
        else: 
            write_frame(frame, frames_dir, resize_width, i)

        success, frame = cap.read()
        i += 1

    cap.release()

    return frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps


# GENERATE VIDEO FROM EXTRACTED FRAMES
def generate_video(frames_dir, fps, token):

    bounded_frames = natsorted(os.listdir(frames_dir))      # get frames sorted by name (1.png to n.png)

    height, width = cv2.imread(os.path.join(frames_dir, bounded_frames[0])).shape[:2]   # get width and height

    video_dir = os.path.join(settings.STATICFILES_DIRS[0], 'results')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    fourcc = cv2.VideoWriter_fourcc(*'vp80')        # set encoder
    video = cv2.VideoWriter(os.path.join(video_dir, token + '.webm'), fourcc, fps, (width, height))       # set result path

    for frame in bounded_frames:        # write frames
        video.write(cv2.imread(os.path.join(frames_dir, frame)))
    
    cv2.destroyAllWindows() 
    video.release()

    return height, width


# RENDERS HOME PAGE FOR DEEPFAKE DETECTION
def index(request):

    videos, placeholders = choose_random_videos()

    media = zip(videos, placeholders)

    context = {
        'media': media
    }

    return render(request, 'deepfake_detection.html', context)


# RENDERS PAGE WITH RESULTS
def predict_video(request):

    token = secrets.token_hex(16)

    if not 'upload' in request.FILES: 
        video_path = request.POST['sample']
        head, video_name = os.path.split(video_path)

        if '_' in video_name:
            with open(os.path.join(settings.VIDEO_ROOT, 'manipulated', video_name), 'rb') as video:
                start_time = datetime.datetime.now().timestamp()
                frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps = extract_and_predict(video, token)
                time = round(datetime.datetime.now().timestamp() - start_time, 2)
            video.close()
        
        else:
            with open(os.path.join(settings.VIDEO_ROOT, 'original', video_name), 'rb') as video:
                start_time = datetime.datetime.now().timestamp()
                frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps = extract_and_predict(video, token)
                time = round(datetime.datetime.now().timestamp() - start_time, 2)
            video.close()

    else:
        video = request.FILES['upload']
        video_name = video.name

        start_time = datetime.datetime.now().timestamp()
        frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps = extract_and_predict(video, token)
        time = round(datetime.datetime.now().timestamp() - start_time, 2)

    height, width = generate_video(frames_dir, fps, token)

    clear_temp(token)

    if inception_predictions != []: 
        average_inception = sum(inception_predictions) / len(inception_predictions)
        average_inception = truncate(100 * average_inception, 2)

        average_xception = sum(xception_predictions) / len(xception_predictions)
        average_xception = truncate(100 * average_xception, 2)

        average_densenet = sum(densenet_predictions) / len(densenet_predictions)
        average_densenet = truncate(100 * average_densenet, 2)

        data_inception = []
        data_xception = []
        data_densenet = []

        for i in range(0, len(inception_predictions)):
            data_inception.append({"x": predictions_indexes[i], "y": inception_predictions[i]})
            data_xception.append({"x": predictions_indexes[i], "y": xception_predictions[i]})
            data_densenet.append({"x": predictions_indexes[i], "y": densenet_predictions[i]})

        context = {
            'result': 'results/' + token + '.webm',
            'time': time,
            'video_name': video_name,
            'width': width, 
            'height': height,
            'average_inception': average_inception,
            'average_xception': average_xception,
            'average_densenet': average_densenet,
            'fps': fps,
            'success': 'true',
            'data_inception': data_inception,
            'data_xception': data_xception,
            'data_densenet': data_densenet,
        }

        return render(request, 'deepfake_detection_prediction.html', context)

    else:
        context = {
            'time': time,
            'video_name': video_name,
            'width': width, 
            'height': height,
            'average_inception': 0.00,
            'average_xception': 0.00,
            'average_densenet': 0.00,
            'fps': fps,
            'success': 'false'
        }

        return render(request, 'deepfake_detection_prediction.html', context)
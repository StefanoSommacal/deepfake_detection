import json
import os
import cv2
import dlib
import shutil
import datetime
import random
import math
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings
from keras.models import load_model
from natsort import natsorted


video_dir = 'http://127.0.0.2:8000/video/'  # video directory on server

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
def clear_temp():

    if os.path.exists(settings.TEMP_ROOT):
        shutil.rmtree(settings.TEMP_ROOT)
    os.makedirs(settings.TEMP_ROOT) 


# CLEAR MEDIA FOLDER
def clear_media():

    if os.path.exists(settings.MEDIA_ROOT):
        shutil.rmtree(settings.MEDIA_ROOT)
    os.makedirs(settings.MEDIA_ROOT)


# CHOOSE 12 RANDOM VIDEOS FROM VIDEO DIRECTORY
def choose_random_videos():

    manipulated = []
    original = []

    with open(os.path.join(settings.VIDEO_ROOT, 'manipulated' , 'manipulated.txt'), 'r') as f:
        manipulated = f.read().splitlines()
    f.close()

    for i in range(len(manipulated)):
        manipulated[i] = video_dir + 'manipulated/' + manipulated[i]

    with open(os.path.join(settings.VIDEO_ROOT, 'original' , 'original.txt'), 'r') as f:
        original = f.read().splitlines()
    f.close()

    for i in range(len(original)):
        original[i] = video_dir + 'original/' + original[i]

    random.shuffle(manipulated)
    random.shuffle(original)

    return manipulated[:6] + original[:6]


# SAVE VIDEO IN TEMP DIR AND RETURNS ITS PATH AND THE DIR PATH
def get_paths(video):

    video_name = os.path.basename(video.name)
    video_name = os.path.splitext(video_name)[0]
    work_dir = os.path.join(settings.TEMP_ROOT, video_name)

    fs = FileSystemStorage(location=work_dir, base_url=work_dir)
    file = fs.save(video_name + '.mp4', video)   # save in media folder
    file = fs.url(file)
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

    if prediction >= 0.75:
        bounded_frame = cv2.rectangle(frame, (x, y), (x+size, y+size), (0,155,0), 2)
        cv2.putText(bounded_frame, 'Original: ' + str(round(prediction*100, 2)) + '%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,155,0), 2)
    elif prediction <= 0.25:
        bounded_frame = cv2.rectangle(frame, (x, y), (x+size, y+size), (0,0,255), 2)
        cv2.putText(bounded_frame, 'Original: ' + str(round(prediction*100, 2)) + '%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    else:
        bounded_frame = cv2.rectangle(frame, (x, y), (x+size, y+size), (43,183,252), 2)
        cv2.putText(bounded_frame, 'Original: ' + str(round(prediction*100, 2)) + '%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (43,183,252), 2)

    return bounded_frame


# EXTRACTS ALL FRAMES FROM THE VIDEO AND RETURNS FRAMES, DIMENSIONS
def extract_frames(video, frames_dir):

    cap = cv2.VideoCapture(video)
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

                face = face / 255
                face = cv2.resize(face, (299, 299), interpolation=cv2.INTER_AREA)
                face = face.reshape(1, 299, 299, channels)
                inception_predictions.append(inception.predict(face)[0][0])
                xception_predictions.append(xception.predict(face)[0][0])
                densenet_predictions.append(densenet.predict(face)[0][0])
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

    return inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps, n_frames


# GET PREDICTION ON EACH FRAME, BOUND FACES AND WRITE FRAME IN A DIRECTORY
def create_prediction(video):

    video_path, work_dir = get_paths(video)

    frames_dir = os.path.join(work_dir, 'frames')       # create directory for frames
    if not os.path.exists(frames_dir):     
        os.makedirs(frames_dir)

    inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps, n_frames = extract_frames(video_path, frames_dir)

    return frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps, n_frames


# GENERATE VIDEO FROM EXTRACTED FRAMES
def generate_video(frames_dir, fps):

    bounded_frames = natsorted(os.listdir(frames_dir))      # get frames sorted by name (1.png to n.png)

    height, width = cv2.imread(os.path.join(frames_dir, bounded_frames[0])).shape[:2]   # get width and height

    fourcc = cv2.VideoWriter_fourcc(*'AVC1')        # set encoder
    video = cv2.VideoWriter(os.path.join(settings.TEMP_ROOT, 'result.mp4'), fourcc, fps, (width, height))       # set result path

    for frame in bounded_frames:        # write frames
        video.write(cv2.imread(os.path.join(frames_dir, frame)))
    
    cv2.destroyAllWindows() 
    video.release()

    fs = FileSystemStorage()
    with open(os.path.join(settings.TEMP_ROOT, 'result.mp4'), 'rb') as video:       # save in media directory
        file = fs.save('result.mp4', video)
        file = fs.url(file)

    return file, height, width


# RENDERS HOME PAGE FOR DEEPFAKE DETECTION
def index(request):

    videos = choose_random_videos()

    context = {
        'videos': videos
    }

    return render(request, 'deepfake_detection.html', context)


# RENDERS PAGE WITH RESULTS
def predict_video(request):

    clear_media()
    clear_temp()

    if not 'uploaded_video_name' in request.FILES: 
        video_path = request.POST['sample']
        video_name = video_path.split('/')[5]

        if '_' in video_name:
            with open(os.path.join(settings.VIDEO_ROOT, 'manipulated', video_name), 'rb') as video:
                start_time = datetime.datetime.now().timestamp()
                frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps, n_frames = create_prediction(video)
                time = round(datetime.datetime.now().timestamp() - start_time, 2)
            video.close()
        
        else:
            with open(os.path.join(settings.VIDEO_ROOT, 'original', video_name), 'rb') as video:
                start_time = datetime.datetime.now().timestamp()
                frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps, n_frames = create_prediction(video)
                time = round(datetime.datetime.now().timestamp() - start_time, 2)
            video.close()

    else:
        video = request.FILES['uploaded_video_name']
        video_name = video.name

        start_time = datetime.datetime.now().timestamp()
        frames_dir, inception_predictions, xception_predictions, densenet_predictions, predictions_indexes, fps = create_prediction(video)
        time = round(datetime.datetime.now().timestamp() - start_time, 2)

    result_path, height, width = generate_video(frames_dir, fps)

    clear_temp()

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
            'time': time,
            'result_path': result_path,
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
            'result_path': result_path,
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
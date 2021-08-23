################################################################################
###  a big thanks to Hemanth Nag for his tutorial:                    ##########
# https://towardsdatascience.com/camera-app-with-flask-and-opencv-bd147f6c0eec #
################################################################################

from flask import Flask, render_template, Response, request
import cv2
import dlib
from PIL import Image
import datetime, time
import os, sys
from matplotlib import pyplot as plt
import numpy as np
from threading import Thread

from app.smiley import Smiley, updateSmiley


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0 # TODO: comment. Just for demo
grey=0 # TODO: delete
neg=0 # TODO: delete
face=0 # will display the smiley
switch=0 # camera switch on/off. We start switch off.
rec=0 # TODO: comment. Just for demo

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

smiley = Smiley()
# fig, ax = plt.subplots()
# fig, (ax1, ax2) = plt.subplots(1, 2)

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.1)
        out.write(rec_frame)


# here, frame is of type numpy.ndarray.
def detect_face(frame):
    data = frame
    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        updateSmiley(smiley, frame)
        plt.figure(figsize=(20,20))
        smiley.draw_smiley(ax1)
        ax2.imshow(frame)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # (h, w) = img.shape
        # print(w,  file=sys.stderr)
        # r = 480 / float(h)
        # dim = ( int(w * r), 480)
        # frame=cv2.resize(img,dim)
        # print('I give the frame!', file=sys.stderr)
    except Exception as e:
        pass
    plt.close(fig)
    return data
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame

    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            # if(grey):
            #     print(type(frame))
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if(neg):
            #     frame=cv2.bitwise_not(frame)    
            # if(capture):
            #     capture=0
            #     now = datetime.datetime.now()
            #     p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            #     cv2.imwrite(p, frame)
            
            # if(rec):
            #     rec_frame=frame
            #     frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            #     frame=cv2.flip(frame,1)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('smiley-template.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        # if request.form.get('click') == 'Capture':
        #     global capture
        #     capture=1
        # elif  request.form.get('grey') == 'Grey':
        #     global grey
        #     grey=not grey
        # elif  request.form.get('neg') == 'Negative':
        #     global neg
        #     neg=not neg
        if  request.form.get('face') == 'Smiley':
            global face
            face=not face 
            if(face):
                time.sleep(4)
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        # elif  request.form.get('rec') == 'Start/Stop Recording':
        #     global rec, out
        #     rec= not rec
        #     if(rec):
        #         now=datetime.datetime.now() 
        #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #         out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
        #         #Start new thread for recording the video
        #         thread = Thread(target = record, args=[out,])
        #         thread.start()
        #     elif(rec==False):
        #         out.release()
                          
                 
    elif request.method=='GET':
        return render_template('smiley-template.html')
    return render_template('smiley-template.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
# cv2.destroyAllWindows()     
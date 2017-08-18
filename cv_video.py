import cv2
import numpy as np
from PIL import Image
import datetime

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def convertToBGR(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cv2pil(cv_img):
    return Image.fromarray(convertToRGB(cv_img))
def pil2cv(pil_img):
    return convertToBGR(np.array(pil_img))

def find_face(img, classifier):
    # Load and convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find face
    faces = classifier.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5);

    # Rectangle around faces
    base = cv2pil(img)
    hats = [None] * len(faces)

    #rect debug
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    """

    base = cv2pil(img)
    #pil processing
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        
        hats[i] = hat.copy()
        hats[i] = hats[i].resize((w*2, h))
        base.paste(hats[i], (x-int(w/2), y), hats[i])

    base = base.transpose(Image.FLIP_LEFT_RIGHT)
        
    return base


#haar = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

hat = Image.open('data/croissant_hair.png')


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()

    img = find_face(frame, lbp_face_cascade)

    cv2.imshow('frame', pil2cv(img))
    c = cv2.waitKey(1)
    if c == ord('q'):
        break
    elif c == ord('s'):
        i = datetime.datetime.now()
        dt = "screenshot-%02d-%02d-%02d" % (i.hour, i.month, i.second)
        img.save('screenshots/'+dt+'.png')
        print("Saved: " + dt)

cap.release()
cv2.destroyAllWindows()

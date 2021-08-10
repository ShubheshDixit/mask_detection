import cv2
import os, time
from tqdm import tqdm

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


def get_face_data(img_type: str):
    face_id = input('\n enter user id end press <return> ==>  ')
    user_name = input('\n enter user name end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    file_count = 0
    pbar = tqdm(total=120)
    while(True):
        ret, frame = cam.read()
        # frame = cv2.flip(frame, -1) # flip video image vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(60, 60)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            file_count += 1
            pbar.update(1)
            os.chdir(os.getcwd())
            if not os.path.exists(f'dataset/{str(face_id)}'):
                os.mkdir(f'dataset/{str(face_id)}')
            if not os.path.exists(f'dataset/{str(face_id)}/{img_type}'):
                os.mkdir(f'dataset/{str(face_id)}/{img_type}')
            cv2.imwrite(f"dataset/{str(face_id)}/{img_type}/{user_name}." + str(face_id) + '.' +  
                        str(file_count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', frame)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            pbar.close()
            break
        elif count >= 120: # Take 360 face sample and stop video
            pbar.close()
            break
    


def work():
    get_face_data('without_mask')
    input("Wear are mask and press <return> ===> ")
    get_face_data('with_mask')
    cam.release()
    cv2.destroyAllWindows()
    # face_training.train_faces()
    # face_recognition_my.start_detection()


if __name__ == '__main__':
    work()
    # get_face_data()
    print("\n [INFO] Exiting Program and cleanup stuff")
    
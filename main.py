import cv2
from ultralytics import YOLO
import cvzone
import math

#model= YOLO('yolov8l.pt')
#p=model.predict(source="111.jpg",save=True)
#print(op)

# For webcam
cap= cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,720)

# For Video
#cap=cv2.VideoCapture("1.mp4")

model=YOLO("yolov8l.pt")

classnames=["person","smartphone","mobile","bottle","car","motorbike","bus","truck","toothbrush","book","cigarette","kettle","train","key","bicycle","aeroplane","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffee","backpack","umbrella","handbag",
            "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","cricket bat","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
            "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","bed","dining table","toilet",
            "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","tooth brush",
            "hair drier"]

while True:
    ok, img=cap.read()
    results=model(img,stream=True)
    # Bounding Boxes
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            #w,h=x2-x1,y2-y1
            #cvzone.cornerRect(img,(x1,y1,w,h))

            #confidence
            conf=math.ceil((box.conf[0]*100))/100

            #class names
            cls= int(box.cls[0])
            cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=3,thickness=3)

    cv2.imshow("image",img)
    cv2.waitKey(1)
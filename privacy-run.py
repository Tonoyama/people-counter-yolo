import cv2
import numpy as np
from PIL import Image

net = cv2.dnn.readNet("./yolov4-tiny.weights", "./yolov4-tiny.cfg")

classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

# 出力レイヤーを定義   
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# 違うクラスごとに違う色を生成する
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)

def main():
    while(True):
        # 読み込み
        ret, frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),(0,0,0),True,crop=False)
        net.setInput(blob)
        results = net.forward(output_layers)

        class_ids=[]
        confidences=[]
        boxes=[]
        confidence_threshold = 0.5

        for result in results:
            for detection in result:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence> confidence_threshold:
                    # オブジェクト検出
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    
                    x = int(center_x-w/2)
                    y = int(center_y-h/2)
                    
                    boxes.append([x,y,w,h]) 
                    confidences.append(float(confidence)) 
                    class_ids.append(class_id) 
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        font = cv2.FONT_HERSHEY_PLAIN
        count = 0
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                color = COLORS[i]

                if int(class_ids[i] == 0):
                    img = cv2.imread("./human.png")
                    back = cv2.imread("./back.png")
                    img_h, img_w, _ = img.shape
                    # img = Image.open("./human.png")
                    # img = img.resize(size=(x, y))
                    a = w/img_w
                    b = h/img_h                    
                    img = cv2.resize(img, None,fx=float(a), fy=float(b))
                    img = Image.fromarray(img)
                    back = Image.fromarray(back)
                    back.paste(img, (x,y))
                    back = np.array(back, dtype=np.uint8)
                    # frame[y:y+h, x:x+w]
                    cv2.rectangle(back, (x,y), (x+w, y+h), color)
                    # cv2.putText(back, label, (x, y-5), font, 1, color, 1)
                    count +=1
        print('Number of people:', count)

        cv2.imshow("Detected_Images",back)
        k = cv2.waitKey(10)
        if k == ord('q'):  break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
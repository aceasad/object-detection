import json
import random
import time
import cv2
import torch
import numpy as np
import mysql.connector

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

label = ["Bottle", "Chair", "Laptop", "Mobile Phone", "Person", "Table"]

basepath_video = 'apiData/video/'
basepath_json = 'apiData/json/'

color = {
    'Bottle': (0, 123, 111),
    'Chair': (222, 111, 212),
    'Laptop': (190, 180, 150),
    'Mobile Phone': (190, 10, 150),
    'Person': (190, 20, 150),
    'Table': (190, 456, 150)
}



class YOLOV5_Detector:
    def __init__(self, weights, img_size, confidence_thres, iou_thresh, agnostic_nms, augment):
        self.weights = weights
        self.imgsz = img_size
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thresh

        self.agnostic_nms = agnostic_nms
        self.augment = augment

        self.device = select_device("")
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size
    
    def checkTableExists(self,dbcon, tablename):
        dbcur = dbcon.cursor()
        dbcur.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{0}'
            """.format(tablename.replace('\'', '\'\'')))
        if dbcur.fetchone()[0] == 1:
            dbcur.close()
            return True

        dbcur.close()
        return False
    def SaveInDB(self, insert_query):
        print("running sql")
        mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="robitacka12",
            database="test"
        )
        table=self.checkTableExists(mydb,"PEGGS")
        c = mydb.cursor()
        if(table == False):
            c.execute("DROP TABLE IF EXISTS PEGGS")
            sql ='''CREATE TABLE PEGGS(
                ID INT AUTO_INCREMENT,
                VIDEO_URL VARCHAR(250),
                JSON_URL VARCHAR(250),
                PRIMARY KEY (ID)
                )'''
            c.execute(sql)
        else:
            insert = "INSERT INTO PEGGS (VIDEO_URL,JSON_URL) VALUES (%s, %s)"
            c.execute(insert, insert_query)
            mydb.commit()

        print(c.rowcount, "record inserted.")
        c.close()
        # query = "insert into video values (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        # columns = ['frame_count', 'millisecond', 'label', 'boundingBoxLeft', 'boundingBoxTop', 'boundingBoxRight', 'boundingBoxBottom', 'confidence']
        # for i, data in df.items():
        #     keys = (i,) + tuple(data[c] for c in columns)
        #     c = mydb.cursor()
        #     c.execute(query, keys)
        #     c.close()

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    def Detect(self, img0):

        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)

        annotation = {}

        count = 1

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    l = label[int(cls.tolist())]
                    c = color[l]
                    annotation["object_" + str(count)] = {}

                    # Add the Object values
                    annotation["object_" + str(count)]['Name'] = l

                    # Add the coordinates value
                    x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    annotation["object_" + str(count)]['X_min'] = x_min
                    annotation["object_" + str(count)]['Y_min'] = y_min
                    annotation["object_" + str(count)]['X_max'] = x_max
                    annotation["object_" + str(count)]['Y_max'] = y_max

                    # Add the confidence value
                    annotation["object_" + str(count)]["Confidence"] = float(conf)

                    #self.plot_one_box(xyxy, img0, color=c, label=l, line_thickness=3)
                    count += 1

        return img0, annotation

    def convert(self,xmin, ymin, xmax, ymax, img_w, img_h):
        dw = 1./(img_w)
        dh = 1./(img_h)
        x = (xmin + xmax)/2.0 - 1
        y = (ymin + ymax)/2.0 - 1
        w = xmax - xmin
        h = ymax - ymin
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return x,y,w,h

    def detect_on_video(self, video_path):
        video_name = video_path.split('.')[0]
        print(video_name)
        jsonFile = basepath_json+video_name+'_labels.json'
        videoFile = basepath_video+video_path

        vid = cv2.VideoCapture(videoFile)
        width  = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        fps = vid.get(cv2.CAP_PROP_FPS)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps = []
        dur_index=0
        allPredictions=[]
        outerJson={'duration':'',"frameRate":fps,"labels":[]}
        for i in range(0,total_frames):
            ret, frame = vid.read()
            if ret:
                res, frame_annot = self.Detect(frame)
                current_frame=int(vid.get(cv2.CAP_PROP_POS_FRAMES))
                timestamps.append(vid.get(cv2.CAP_PROP_POS_MSEC))                
                print("Current Frame: ",current_frame," / ",total_frames," at miliseconds ",timestamps[i])
                
                for k in range (0,len(frame_annot)):
                    index=str(k+1)
                    #convert points to percentages 
                    label=frame_annot['object_'+index]['Name']
                    x_min=frame_annot['object_'+index]['X_min']
                    y_min=frame_annot['object_'+index]['Y_min']
                    x_max=frame_annot['object_'+index]['X_max']
                    y_max=frame_annot['object_'+index]['Y_max']
                    x,y,w,h=self.convert(x_min,y_min,x_max,y_max,width,height)

                    confidence=round(frame_annot['object_'+index]['Confidence']* 100, 3)
                    
                    #Storing in Json
                    if(timestamps[i] > 0 and i>0): 
                        data={'confidence':confidence,'label':label,'boundingBoxHeight':h,'boundingBoxLeft':x,'boundingBoxTop':y,'boundingBoxWidth':w,'miliseconds':int(round(timestamps[i],1))}
                        allPredictions.append(data)
                        dur_index=i
                #if "millisecond-"+str(timestamps[i]) not in json_res:
                #    json_res.setdefault(str(timestamps[i]),outerJson)
                i=current_frame

        outerJson['labels']=allPredictions
        outerJson['duration']=int(round(timestamps[dur_index],1))
        vid.release()
        cv2.destroyAllWindows()
        #Append the ID for table with labels
        json.dump(outerJson, open(jsonFile, 'w'))
        val = (videoFile, jsonFile)
        self.SaveInDB(val)
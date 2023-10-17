import os
import random
import torch
import re, warnings
import requests
from dict_mapping import URL_DICT, URL_ID_DICT
from PIL import Image
import pandas as pd

from super_gradients.training import models



class config:
    # trainer params
    CHECKPOINT_DIR = 'checkpoints'  # specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'finding-battleships'  # specify the experiment name

    CLASSES = ['Prescription-Pharmacist', 'Reviewer', 'Dispenser']

    NUM_CLASSES = len(CLASSES)

    # model params
    MODEL_NAME = 'yolo_nas_l'  # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco'  # only one option here: coco



model = models.get(config.MODEL_NAME,
                   num_classes=config.NUM_CLASSES,
                   pretrained_weights=config.PRETRAINED_WEIGHTS
                   )


best_model = models.get(config.MODEL_NAME,
                        num_classes=config.NUM_CLASSES,
                        checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))


def run(val_image_root_path):

    ID = []
    IMAGE_NAME = []
    IS_ORIGINPICNAME = []
    IS_AUDITPICNAME = []
    Prescription_Pharmacist_coordinates = []
    Reviewer_coordinates = []
    Dispenser_coordinates = []
    

    for imgCls in os.listdir(val_image_root_path):

        if not os.path.exists(f'./PredImgRes/{imgCls}'):
            os.makedirs(f'./PredImgRes/{imgCls}')
            
        for imgPath in os.listdir(f'./OracleImg/{imgCls}/'):
            
            save_name_dir = imgPath.split('.')[0].split('/')[-1]
            predict_res = best_model.predict(f'./OracleImg/{imgCls}/{imgPath}', conf=0.4)
        
            predict_res.save(f"./PredImgRes/{imgCls}/{save_name_dir}")
            
            with open('DetectionResult.txt', 'w') as f:
                f.write(str(predict_res))
            
            with open('DetectionResult.txt', 'r') as file:
                data = file.read()
        
            start_str = "prediction=DetectionPrediction("
            end_str = ")])"
            
            start_index = data.find(start_str)
            end_index = data.find(end_str)
            
            input_string = data[start_index + len(start_str):end_index]
            input_string = input_string.replace('\n', ' ')
            
        
            bboxes_match = re.search(r'bboxes_xyxy=array\((.*?)\),', input_string)
            confidence_match = re.search(r'confidence=array\((.*?)\),', input_string)
            labels_match = re.search(r'labels=array\((.*?)\),', input_string)
            class_names_match = re.search(r'class_names=\[(.*?)\]', input_string)
            
            if bboxes_match and confidence_match and labels_match and class_names_match:
                bboxes_info = bboxes_match.group(1)
                confidence_info = confidence_match.group(1)
                labels_info = labels_match.group(1)
                class_names_info = class_names_match.group(1)
            
                with open('PicPrediction.txt', 'w') as f:
                    f.write(f"bboxes_xyxy: {re.sub(r', dtype=float32|, dtype=float16|    ', '', bboxes_info)}" + '\n')
                    f.write(f"confidence: {re.sub(r', dtype=float32|, dtype=float16|    ', '', confidence_info)}" + '\n')
                    f.write(f"labels: {re.sub(r', dtype=float32|, dtype=float16|    |', '', labels_info)}" + '\n')
                    f.write(f"class_names: {re.sub(r', dtype=float32|, dtype=float16|    ', '', class_names_info)}" + '\n')
            else:
                print("No match found")
            
            with open('PicPrediction.txt', 'r') as file:
                lines = file.readlines()
        
            bboxes_xyxy = []
            confidence = []
            labels = []
            class_names = []

            try:
                for line in lines:
                    if line.startswith("bboxes_xyxy:"):
                        bboxes_xyxy_str = line.replace("bboxes_xyxy:", "").strip()
                        bboxes_xyxy = eval(bboxes_xyxy_str)
                
                    elif line.startswith("confidence:"):
                        confidence_str = line.replace("confidence:", "").strip()
                        confidence = eval(confidence_str)
                
                    elif line.startswith("labels:"):
                        labels_str = line.replace("labels:", "").strip()
                        labels_str = labels_str.replace(")", "").strip()
                        labels = eval(labels_str)
                
                    elif line.startswith("class_names:"):
                        class_names_str = line.replace("class_names:", "").strip()
                        class_names = eval("[" + class_names_str + "]")
            except Exception as e:
                with open('ImgPredictFailed.txt', 'a') as f:
                    f.write(URL_DICT[imgPath] + '\n' + '\n')

            # Write the result to dataframe
            if len(labels) == 3:
                # convert to set avoid to happen repeat bboxes
                if len(set(labels)) == 3:
                    if imgCls == 'ORIGINPICNAME':
                        IS_ORIGINPICNAME.append(1)
                        IS_AUDITPICNAME.append(0)
                    else:
                        IS_ORIGINPICNAME.append(0)
                        IS_AUDITPICNAME.append(1)
    
                    IMAGE_NAME.append(URL_DICT[imgPath])
                    ID.append(URL_ID_DICT[imgPath])
                        
                    for idx in range(len(bboxes_xyxy)):
                        x_label = int(bboxes_xyxy[idx][2])
                        y_label = int(((bboxes_xyxy[idx][3] - bboxes_xyxy[idx][1]) / 2) + bboxes_xyxy[idx][1])
                        class_info = class_names[int(labels[idx])]
        
                        if 'Prescription-Pharmacist' in class_info:
                            Prescription_Pharmacist_coordinates.append([x_label, y_label])
                        elif 'Reviewer' in class_info:
                            Reviewer_coordinates.append([x_label, y_label])
                        elif 'Dispenser' in class_info:
                            Dispenser_coordinates.append([x_label, y_label])
                else:
                    with open('ImgLogInfo.txt', 'a') as f:
                        f.write(f"Number of bboxes: {len(bboxes_xyxy)}, root is: '/usr/local/Huangshuqi/jupterCode/Obj_detection/PredImgRes/{imgCls}/{save_name_dir}'" + '\n' + '\n')
            else:
                pass
                # print(f'The class info is {class_info}, and Bounding box coordinates of colon syntax is {(x_label, y_label)}')
    
    
    # print('Length of ID: ', len(ID))
    # print('Length of IMAGE_NAME: ', len(IMAGE_NAME))
    # print('Length of IS_ORIGINPICNAME: ', len(IS_ORIGINPICNAME))
    # print('Length of IS_AUDITPICNAME: ', len(IS_AUDITPICNAME))
    # print('Length of Prescription_Pharmacist_coordinates: ', len(Prescription_Pharmacist_coordinates))
    # print('Length of Reviewer_coordinates: ', len(Reviewer_coordinates))
    # print('Length of Dispenser_coordinates: ', len(Dispenser_coordinates))
    
    dfRes = pd.DataFrame({'REMOTEAUDITID': ID, 'IMAGE_NAME': IMAGE_NAME, 
                          'IS_ORIGINPICNAME': IS_ORIGINPICNAME, 'IS_AUDITPICNAME': IS_AUDITPICNAME,
                         'Prescription_Pharmacist_coordinates': Prescription_Pharmacist_coordinates,
                         'Reviewer_coordinates': Reviewer_coordinates,
                         'Dispenser_coordinates': Dispenser_coordinates})
    dfRes.to_csv('./ImgPredictRes.csv', index=False)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--val_image_root_path', required=True, help='The directory of your image root path')
#     args = parser.parse_args()
#     run(args)
    
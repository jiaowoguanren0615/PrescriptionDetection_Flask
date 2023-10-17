import os, re, requests
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



def GetPicTure(image_url):
    if not os.path.exists('./save_img'):
        os.makedirs('./save_img')
    try:
        img_response = requests.get(image_url)

        if img_response.status_code == 200:
            file_name = image_url.split('/')[-1]
            save_path = os.path.join('./save_img', file_name)
            with open(save_path, 'wb') as img_file:
                img_file.write(img_response.content)
            return save_path
        else:
            with open('./DownLoadFailed.txt', 'a') as f:
                f.write(image_url + '\n')
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
    return False


def get_prediction(val_image_root_path):
    Prescription_Pharmacist_coordinates = []
    Reviewer_coordinates = []
    Dispenser_coordinates = []
    predict_res = best_model.predict(val_image_root_path, conf=0.4)

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
            f.write(
                f"confidence: {re.sub(r', dtype=float32|, dtype=float16|    ', '', confidence_info)}" + '\n')
            f.write(f"labels: {re.sub(r', dtype=float32|, dtype=float16|    |', '', labels_info)}" + '\n')
            f.write(
                f"class_names: {re.sub(r', dtype=float32|, dtype=float16|    ', '', class_names_info)}" + '\n')
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
            f.write(f'Detection failed, the image url is: {val_image_root_path}' + '\n' + '\n')

    # Write the result to dataframe
    if len(labels) == 3:
        # convert to set avoid to happen repeat bboxes
        if len(set(labels)) == 3:
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

            return_info = {
                'Image website': val_image_root_path,
                'Prescription-Pharmacist coordinates': Prescription_Pharmacist_coordinates,
                'Review-Pharmacist coordinates': Reviewer_coordinates,
                'Distribution-Pharmacist coordinates': Dispenser_coordinates
            }
        else:
            return_info = {'Image website': val_image_root_path, "result": ['Can not detect three coordinates']}
    else:
        return_info = {'Image website': val_image_root_path, "result": ['Can not detect three coordinates']}
    return return_info

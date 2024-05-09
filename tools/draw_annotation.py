from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os

char_to_thai = {'0':'0' , '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9'
                , 'c1':'ก', 'c2':'ข', 'c3':'ฃ', 'c4':'ค', 'c5':'ฅ', 'c6':'ฆ', 'c7':'ง', 'c8':'จ', 'c9':'ฉ', 'c10':'ช'
                , 'c11':'ซ', 'c12':'ฌ', 'c13':'ญ', 'c14':'ฎ', 'c15':'ฏ', 'c16':'ฐ', 'c18':'ฒ', 'c19':'ณ', 'c20':'ด'
                , 'c21':'ต', 'c22':'ถ', 'c23':'ท', 'c24':'ธ', 'c25':'น', 'c26':'บ', 'c27':'ป', 'c28':'ผ', 'c30':'พ'
                , 'c31':'ฟ', 'c32':'ภ', 'c33':'ม', 'c34':'ย', 'c35':'ร', 'c36':'ล', 'c37':'ว', 'c38':'ศ', 'c39':'ษ'
                , 'c40':'ส', 'c41':'ห', 'c42':'ฬ', 'c43':'อ', 'c44':'ฮ'}

colour = {'red':(57, 17, 217),'blue':(204, 102, 0),'orange':(4, 125, 246),'pink':(183, 55, 249),'green':(38, 89, 2),'yellow':(19, 173, 209),'purple':(200, 58, 133),'neon':(179, 190, 101) }

# for i in colour.values :
#     print(i)

# if the folder is not exists, create first
save_folder_picture = "./data_save/picture/"
if not os.path.exists(save_folder_picture):
    os.makedirs(save_folder_picture)
    print(f"Created folder: {save_folder_picture}")

save_folder_labels = "./data_save/labels/"
if not os.path.exists(save_folder_labels):
    os.makedirs(save_folder_labels)
    print(f"Created folder: {save_folder_labels}")

# Define the function
def yolo_anotation(list_name, model_path):
    model = YOLO(model_path)  # Load the model
    for image_name in [list_name]:  # Loop all the pictures in the list
        list_data = []  # Create the list for annotation
        list_annotation = []
        
        images = cv2.imread(image_name)  # Load image with opencv
        resized_image = cv2.resize(images, (640, 640))  # Resize the image before sending to model
        results = model.predict(resized_image, conf=0.4)  # Detect the character with confidence 0.4
        annotator = Annotator(resized_image, line_width=5, example=str(char_to_thai))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # b = box.xyxy[0] / 640
                b = box.xyxy[0]
                c = int(box.cls)
                label = char_to_thai.get(model.names[c], model.names[c])  # Use char_to_thai dict or model name
                result_string = ' '.join(map(str, b.cpu().tolist()))
                list_data.append(f'{label} {result_string}')
                list_annotation.append(box.xyxy[0])
                # annotator.box_label(box.xyxy[0], color=colour['red'])
        sorted_tensors = sorted(list_annotation, key=lambda x: x[0])
        # for i in sorted_tensors :
        #     print(np.array)
        for anto, coloue in zip(sorted_tensors, colour.values()) :
            annotator.box_label(anto, color=coloue)
        sorted_elements = sorted(list_data, key=lambda x: float(x.split()[1]))
        real_name = os.path.splitext(os.path.basename(image_name))[0]  # Get the real name

        annotated_image = annotator.result()
        cv2.imwrite(f'{save_folder_picture}/{real_name}.jpg', annotated_image)

        with open(f'{save_folder_labels}/{real_name}.txt', 'wb') as file:
            file.write('\n'.join(sorted_elements).encode('utf-8'))
    return f'{save_folder_picture}/{real_name}.jpg'


if __name__ == '__main__' :
    yolo_anotation(['data_save/license_plate/license_plate_save.jpg'], './digit.pt')
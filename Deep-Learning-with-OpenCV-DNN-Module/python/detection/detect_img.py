import cv2
import numpy as np

# se incarca numele clasei COCO
with open('../../input/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# se obtine diferite vectori de culoare pentru fiecare clasa
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# se incarca modelul DNN
model = cv2.dnn.readNet(model='../../input/frozen_inference_graph.pb',
                        config='../../input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

# se citeste imaginea de pe disk
image = cv2.imread('biciclete.png')
image_height, image_width, _ = image.shape
# se creaaza un blob pentru imagine
blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123),
                             swapRB=True)
model.setInput(blob)
# se trece la pasul următor pentru efectuarea detectiei
output = model.forward()

# se realizeaza o bucla for pentru detectarea fiecarui obiect din imagine
for detection in output[0, 0, :, :]:
    # se extrage confidenta pentru ficare detectie
    confidence = detection[2]
    # se desenează un chenar pentru fiecare obiect detectat.
    if confidence > .2:
        # se obtine id-ul clasei
        class_id = detection[1]
        # se mapeaza clasa cu id-ul ei
        class_name = class_names[int(class_id)-1]
        color = COLORS[int(class_id)]
        # se obtine coordonatele tuturor chenarelor
        box_x = detection[3] * image_width
        box_y = detection[4] * image_height
        # se obtine lungimea si latimea fiecarui chenar
        box_width = detection[5] * image_width
        box_height = detection[6] * image_height
        # se deseneaza un patrulater fiecarui obiect desenat
        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
        # se aduga text deasupra fiecarui patrulater
        cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow('image', image)
cv2.imwrite('marcate.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

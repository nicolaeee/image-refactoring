import cv2
import numpy as np
# citim clasa ImageNet
with open('../../input/classification_classes_ILSVRC2012.txt', 'r') as f:
    image_net_names = f.read().split('\n')
# numele clasei finale ( doar primul cuvânt din numeroasele nume ImageNet pentru o imagine)
class_names = [name.split(',')[0] for name in image_net_names]
# încarcam modelul network
model = cv2.dnn.readNet(model='../../input/DenseNet_121.caffemodel',
                      config='../../input/DenseNet_121.prototxt',
                      framework='Caffe')
# incarcam imaginea de pe disc
image = cv2.imread('m8.jpeg')
# punem imaginea in evidenta
blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224),
                             mean=(104, 117, 123))
# setam ca intrare peunerea in evidenta pentru retea
model.setInput(blob)
# trimitem imaginea sa treaca prin model
outputs = model.forward()
final_outputs = outputs[0]
# marcam toate iesirile
final_outputs = final_outputs.reshape(1000, 1)
# obtinem eticheta clasei
label_id = np.argmax(final_outputs)
# convertim scorul de iesire cu functia de probabilitate  softmax
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
# obtinem ce mai mare probabilitate
final_prob = np.max(probs) * 100.
# mapați încrederea maximă a etichetelor de clasă
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"
# punem numele claselor sus pe imagine
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('image.jpg', image)

import cvzone
import cv2
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
myClassifier = cvzone.Classifier('C:/Users/Radhitya Harun/Documents/kuliah/Semester 5/Sistem Cerdas/Projek SC (2)/Projek SC/Model/keras_model.h5', 'C:/Users/Radhitya Harun/Documents/kuliah/Semester 5/Sistem Cerdas/Projek SC (2)/Projek SC/Model/labels.txt')
fpsReader = cvzone.FPS()


while True :
	_, img = cap.read()
	prediction, index = myClassifier.getPrediction(img, scale=1.5)
	#print(prediction, index)
	fps, img = fpsReader.update(img, pos=(50, 100))
	print(fps)

	cv2.imshow("Image", img)
	cv2.waitKey(1)



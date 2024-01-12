import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from skimage.feature import hog
from mlxtend.data import loadlocal_mnist

images, labels = loadlocal_mnist(
    images_path= r'C:\Users\ASUS\Pictures\mnist-dataset/train-images-idx3-ubyte',
    labels_path= r'C:\Users\ASUS\Pictures\mnist-dataset/train-labels-idx1-ubyte'
)

hog_features = [
    hog(image.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)[0]
    for image in images[:100]
]


hog_features = np.array(hog_features)

X_train, X_test, y_train, y_test = train_test_split(hog_features, labels[:100], test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
print(hog_features, 'hog features')

y_pred = svm_classifier.predict(X_test)

result_pairs = list(zip(y_test, y_pred))
print("True Label  |  Predicted Label")
print("----------------------------")
for true_label, predicted_label in result_pairs:
    print(f"{true_label:11} | {predicted_label:15}")

akurasi = accuracy_score(y_test, y_pred)
presisi = precision_score(y_test, y_pred, average='weighted', zero_division=1)
confussio_matrix = confusion_matrix(y_test, y_pred)
print("\nEvaluation Metrics:")
print(akurasi)
print(presisi)
print(confussio_matrix)
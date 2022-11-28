import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

def read_txt_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    return lines

def extract_image_and_label(line: str) -> tuple[np.ndarray, int]:

    label = int(line[-2])

    vector = list(
        map(lambda x: int(x), line[:-3].split(" "))
    )

    return vector, label

def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    lines: list[str] = read_txt_file('./datasets/ocr_car_numbers_rotulado.txt')
    X = []
    Y = []
    for line in lines:
        (x, y) = extract_image_and_label(line)
        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)

def show_image(image_vector: list[int], label: int):

    image: np.array = np.reshape(image_vector, (35, 35))

    plt.imshow(image)
    plt.show()

def split_data(X, Y, test_size: int = 0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    return (X_train, X_test, y_train, y_test)

if __name__ == '__main__':

    X, Y = load_dataset()
    (X_train, X_test, y_train, y_test) = split_data(X, Y)
    print(X_train.shape, X_test.shape)
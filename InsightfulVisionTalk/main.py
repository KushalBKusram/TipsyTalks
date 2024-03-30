import cv2
import pytesseract
import numpy as np

def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is not None:
        # Display the image
        return image
    else:
        print('Error: Unable to read the image.')

def detect_edge(image: np.ndarray) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200)
    return edges

def detect_corners(image: np.ndarray) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(grayscale_image, 2, 3, 0.04)
    # Threshold for selecting strong corners
    threshold = 0.01 * corners.max()
    # Draw circles around the detected corners
    image[corners > threshold] = [0, 0, 255]
    return image

def detect_face(image: np.ndarray) -> np.ndarray:
    # Load the pre-trained cascade classifier for detecting faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def detect_text(image: np.array) -> str:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image (optional)
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(threshold_image)
    # Print the extracted text
    return text


if __name__=="__main__":

    image_path = "data/shapes.png"
    lena_path = "data/lena.png"
    text_path = "data/ocr.png"

    image = read_image(image_path)
    cv2.imshow("Shapes - Image", image)
    edges = detect_edge(image)
    cv2.imshow("Shapes - Edges", edges)
    corners = detect_corners(image)
    cv2.imshow("Shapes - Corners", corners)

    image = read_image(lena_path)
    face_detected = detect_face(image)
    cv2.imshow("Lena", face_detected)
    
    image = read_image(text_path)
    cv2.imshow("Text", image)
    text = detect_text(image)
    print(f"Here is the text from the image: {text}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame (optional, to get mirror image)
    frame = cv2.flip(frame, 1)

    # Convert to HSV (Hue, Saturation, Value) for easier color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting skin color (adjust based on your lighting/environment)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the result to grayscale and blur it
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to get a binary image (just white and black)
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours (used for detecting gestures)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours by area and get the largest one (potential hand)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        hand_contour = contours[0]

        # Get the convex hull of the hand contour
        hull = cv2.convexHull(hand_contour)

        # Draw the contours and hull on the frame
        cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

        # Approximate the convex hull to identify the number of fingers
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)

        # If there are enough points, count fingers
        if len(approx) > 5:
            cv2.putText(frame, "Gesture: Open Hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Gesture: Fist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

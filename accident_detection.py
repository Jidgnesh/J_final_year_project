from keras.models import load_model
import cv2
import numpy as np
import time
from twilio.rest import Client
import socket

# Set up Twilio client for SMS notification
TWILIO_SID = 'ACdafdc5b0e63e82c7639926781db2f8f0'
TWILIO_AUTH_TOKEN = 'e6e801d78b0c3ed80a3df67b35572c9f'
TWILIO_PHONE_NUMBER = '+15674234597'
EMERGENCY_PHONE_NUMBER = '+919886762386'
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Load the model from the h5 file
#

# Set up OpenCV video capture
cap = cv2.VideoCapture('v13.mp4')

# # get the hostname
# hostname = socket.gethostname()
# # get the IP address
# ip_address = socket.gethostbyname(hostname)

# Define a function to send an SMS notification using Twilio
def send_sms_notification():
    message = client.messages.create(
        # body=f'Vehicle accident detected at {ip_address}. Please send emergency services.',
        body='Vehicle accident detected, Please send emergency services.',
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_PHONE_NUMBER
    )
    print('SMS notification sent.')

fgbg = cv2.createBackgroundSubtractorMOG2()

# Flag to indicate if an accident has been detected
accident_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)

    thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]
    

    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # adjust the threshold as needed
            # Set the accident_detected flag to True
            accident_detected = True
            send_sms_notification()
            

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Accident Detection', frame)
    
 
    if accident_detected:
        time.sleep(7)  
        accident_detected = False
    
    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()

import cv2
import sys
import os

class captureCameraDrawLineRecordWidget(object):
    def captureCameraView(self):
        source = "rtsp://admin:AdminAdmin1@192.168.1.126:554/cam/realmonitor?channel=1&subtype=1"
        cap = cv2.VideoCapture(source,cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print('Cannot open RTSP stream')
            exit(0)

        # Step 1: Capture and save the first frame from the camera.
        while True:
            ret, frame = cap.read()
            if not ret:
                print ("Can\'t get frame")
                exit(0)
            break
           
        cv2.imwrite("samples/CaptureCameraFirstView_v1.png", frame)
        
    def __init__(self):
        self.captureCameraView()
        self.original_image = cv2.imread('samples/CaptureCameraFirstView_v1.png')
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone) 
            
            output_file = open("samples/coordinates_v1.txt","w")
            output_file.write('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            output_file.close()

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

if __name__ == '__main__':
    widget = captureCameraDrawLineRecordWidget()
    while True:
        cv2.imshow('image', widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
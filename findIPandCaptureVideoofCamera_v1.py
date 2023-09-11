# Step 1: Sedarch IP address of IP cameras in the same network with PC.
# Step 2: Retrieve view of IP cameras for corresponding IP address
# Step 3: Upload the raw view of IP cameras to cloud API
 
import cv2
import sys
import os
import nmap

class findIPAddressAndCaptureViewofCameraWidget(object):
    def findIPAddressAndCatchCameraView(self):
        # Step 0: Search the same network and find available IP address 
        output_file = open("samples/cameras.yml","w")
        nm = nmap.PortScanner()
        results = nm.scan(hosts='192.168.1.1/24', arguments='-sP')
        # print (results)
        print (results['scan'])
        IP_List = list(results['scan'].keys())
        print (IP_List)
        print (results['scan']['192.168.1.126']['vendor'])

        IP_address = '0.0.0.0'
        # Loop the IP_list
        for IP in IP_List:
            if IP == '192.168.1.1':	# Skip the local network IP address 
                continue
            else:
                print (results['scan'][IP]['vendor'])

                if 'Amcrest Technologies' in str(results['scan'][IP]['vendor']):
                    IP_address = IP
                    break

        # print ('IP_address:' + IP_address)
        # output_file.write('IP_address:' + IP_address)

        line = 'cameras:\n'
        output_file.write(line)
        line = '  IP_address: ' + IP_address + '\n'
        output_file.write(line)
        output_file.close()
        
        # Step 1: Catch the camera's view based on IP address and password
        user_id = "admin"
        password = "AdminAdmin1"
        rtsp_address = "rtsp://" + user_id + ":" + password + "@" + IP_address + ":" + "554/cam/realmonitor?channel=1&subtype=1"
        print (rtsp_address)
       
        # source = "rtsp://admin:AdminAdmin1@192.168.1.126:554/cam/realmonitor?channel=1&subtype=1"
        source = rtsp_address
        cap = cv2.VideoCapture(source,cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print('Cannot open RTSP stream')
            exit(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print ("Can\'t get frame")
                exit(0)
            break
           
        cv2.imwrite("samples/CaptureCameraFirstView_v1.png", frame)
        
        window_name = 'image'
  
        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow(window_name, frame)
  
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
  
        # closing all open windows
        cv2.destroyAllWindows()
        
    def __init__(self):
        self.findIPAddressAndCatchCameraView()
        
if __name__ == '__main__':
    widget = findIPAddressAndCaptureViewofCameraWidget()
    
   

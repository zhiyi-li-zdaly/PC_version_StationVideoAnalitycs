# Version 2: Add User Interface 
# Step 1: Sedarch IP address of IP cameras in the same network with PC.
# Step 2: Retrieve view of IP cameras for corresponding IP address
# Step 3: Upload the raw view of IP cameras to cloud API
 
import cv2
import sys
import os
import nmap
import PySimpleGUI as sg

class findIPAddressAndCaptureViewofCameraWidget(object):
    def findIPAddressAndCatchCameraView(self):
        # Step 0: Search the same network and find available IP address 
        output_file = open("samples/cameras.yml","w")
        nm = nmap.PortScanner()
        results = nm.scan(hosts='192.168.1.1/24', arguments='-sP')
        # print (results)
        print (results['scan'])
        
        # Add User Interface to demonstrate available IP cameras.

        IP_List = list(results['scan'].keys())
        
        print (IP_List)
        
        IP_vendor_List = []
        for IP in IP_List:
            vendor = str(results['scan'][IP]['vendor'])
            print (vendor)
            IP_vendor = IP + "," + vendor
            IP_vendor_List.append(IP_vendor)
        
        print (IP_vendor_List)

        # key="-FOLDER-" folder = values["-FOLDER-"]
        lst1 = sg.Combo(IP_vendor_List, font=('Arial Bold', 14),  expand_x=True, enable_events=True,  readonly=False, key='-lst1-', visible=False)
        lst2 = sg.Combo(IP_vendor_List, font=('Arial Bold', 14),  expand_x=True, enable_events=True,  readonly=False, key='-lst2-', visible=False)
        lst3 = sg.Combo(IP_vendor_List, font=('Arial Bold', 14),  expand_x=True, enable_events=True,  readonly=False, key='-lst3-', visible=False)
        lst4 = sg.Combo(IP_vendor_List, font=('Arial Bold', 14),  expand_x=True, enable_events=True,  readonly=False, key='-lst4-', visible=False)
        
        device_nums = []
        device_nums.append(1)
        device_nums.append(2)
        device_nums.append(3)
        device_nums.append(4)
        dst = sg.Combo(device_nums, font=('Arial Bold', 14), expand_x=True, enable_events=True,  readonly=False, key='-COMBO-')
         
        layout = [[sg.Text("Enter number of devices"), dst, sg.Button("Select")], [sg.Text("Select devices from the list")], [lst1], [lst2], [lst3], [lst4], [sg.Cancel(),  sg.Submit()]]
        
        # Create the window
        window = sg.Window("Device IP Selection Wizard", layout, size=(715, 400))

        IP_vendor_val_List = []
        # Create an event loop
        while True:
            event, values = window.read()
            if event == "Select": 
                nums = dst.get()
                print ("nums: ", dst.get())
                
                if dst.get() == 1:
                    window["-lst1-"].update(visible=True)
                    
 
                if dst.get() == 2:
                    window["-lst1-"].update(visible=True)
                    window["-lst2-"].update(visible=True)
               
                if dst.get() == 3:
                    window["-lst1-"].update(visible=True)
                    window["-lst2-"].update(visible=True)
                    window["-lst3-"].update(visible=True)
 
                if dst.get() == 4:
                    window["-lst1-"].update(visible=True)
                    window["-lst2-"].update(visible=True)
                    window["-lst3-"].update(visible=True)
                    window["-lst4-"].update(visible=True)

            # Grab value from 
            # End program if user closes window or
            # presses the OK button
            if event == "Submit": 
                num = dst.get()
                if dst.get() == 1:
                    IP_vendor_val = lst1.get()
                    IP_vendor_val_List.append(IP_vendor_val)

                    
 
                if dst.get() == 2:
                    IP_vendor_val = lst1.get()
                    IP_vendor_val_List.append(IP_vendor_val)
                    IP_vendor_val = lst2.get()
                    IP_vendor_val_List.append(IP_vendor_val)                    

                if dst.get() == 3:
                    IP_vendor_val = lst1.get()
                    IP_vendor_val_List.append(IP_vendor_val)
                    IP_vendor_val = lst2.get()
                    IP_vendor_val_List.append(IP_vendor_val) 
                    IP_vendor_val = lst3.get()
                    IP_vendor_val_List.append(IP_vendor_val)
 
                if dst.get() == 4:
                    IP_vendor_val = lst1.get()
                    IP_vendor_val_List.append(IP_vendor_val)
                    IP_vendor_val = lst2.get()
                    IP_vendor_val_List.append(IP_vendor_val) 
                    IP_vendor_val = lst3.get()
                    IP_vendor_val_List.append(IP_vendor_val)
                    IP_vendor_val = lst4.get()
                    IP_vendor_val_List.append(IP_vendor_val)
                   
                print (IP_vendor_val_List)
                    
                break
            if event == sg.WIN_CLOSED or event == 'Cancel' or event == None:
                break

        window.close()
  
        print ("Test: ", IP_vendor_val_List)
        if len(IP_vendor_val_List) == 0:
            print ("No cameras were found: ")
            exit(0)
        
        IP_address = IP_vendor_val_List[0].split(",")[0]
        print (IP_address)

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
    
   

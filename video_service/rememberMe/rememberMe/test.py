# Python script to generate test base64 image
import base64

# Read an image file
with open('/Users/swagatbhowmik/CS projects/rtMe/video_service/rememberMe/AC35C2AF-D551-48D8-BB2A-81CD137C326C.JPG', 'rb') as image_file:
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')

print(base64_string)

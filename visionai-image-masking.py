
# Import the necessary libraries
import cv2 # OpenCV library for computer vision
import numpy as np # NumPy library for numerical operations
import urllib.request # urllib library for handling URLs

# Import the Azure AI Vision SDK
import azure.ai.vision as sdk

# Create a service options object with the endpoint and key of the Azure Vision service
service_options = sdk.VisionServiceOptions("your cognative searc URL API")

# Define the URL of the image to be analyzed
mask_image_url =  "the image u would like to vision ai to read"

# Create a vision source object with the image URL
vision_source = sdk.VisionSource(
    url=mask_image_url)

# Create an image analysis options object with the model name
analysis_options = sdk.ImageAnalysisOptions()
analysis_options.model_name = "testingtrainer"

# Create an image analyzer object with the service options, vision source, and analysis options
image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

# Analyze the image and get the result
result = image_analyzer.analyze()

# Check if the result is valid
if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

    # Check if there are any custom objects detected in the image
    if result.custom_objects is not None:
        print(" Custom Objects:")
        # Loop through each custom object
        for object in result.custom_objects:
            # Print the name, bounding box, and confidence of the object
            # print("   '{}', {} Confidence: {:.4f}".format(object.name, object.bounding_box, object.confidence))
            # Check if the object is a HKID card and has a high confidence score
            if object.name == "HKID" and object.confidence >= 0.8:
                # Get the image from the URL
                url = mask_image_url
                img = urllib.request.urlopen(url)
                img_array = np.array(bytearray(img.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Define the region of interest (ROI) as the bounding box of the HKID card
                roi = (object.bounding_box.x, object.bounding_box.y, object.bounding_box.w, object.bounding_box.h)

                # Create a binary mask with the same dimensions as the original image
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

                # Set all pixels inside the ROI to black
                mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 0

                # Apply the mask to the original image to hide the HKID card details
                masked_img = cv2.bitwise_and(image, image, mask=mask)

                # Display the masked image
                cv2.imshow('Masked Image', masked_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
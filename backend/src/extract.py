import boto3

# The path to your image file on your device
local_image_path = 'E:/Bajaj motor insurance/Backend/images/MedicalBill.jpg' 

# Create a Textract client
# Ensure your AWS credentials are configured (e.g., via `aws configure`)
try:
    textract_client = boto3.client('textract')

    # Read the image file as bytes
    with open(local_image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # Call the Textract API
    # We use 'Bytes': image_bytes to send the local file
    response = textract_client.detect_document_text(
        Document={'Bytes': image_bytes}
    )

    # Print the detected text
    print("--- Detected Text ---")
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            print(item["Text"])
    print("---------------------")

except FileNotFoundError:
    print(f"Error: The file '{local_image_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
import os
import requests

def download_image(image_url, save_directory="images"):
  """
  Downloads an image from the specified URL and saves it to the given directory.

  Args:
      image_url (str): The URL of the image to download.
      save_directory (str, optional): The directory to save the image to. Defaults to "images".

  Returns:
      str: The path to the downloaded image file, or None if an error occurs.
  """

  # Check if save directory exists, create it if not
  if not os.path.exists(save_directory):
    os.makedirs(save_directory)

  # Extract filename from URL or use a default name
  filename = os.path.basename(image_url)
  if not filename:
    filename = "image.jpg"  # Default filename

  # Construct full save path
  save_path = os.path.join(save_directory, filename)

  try:
    # Send HTTP GET request to download the image
    response = requests.get(image_url, stream=True)

    # Check for successful response (status code 200)
    if response.status_code == 200:
      # Open the file in binary write mode
      with open(save_path, "wb") as f:
        for chunk in response.iter_content(1024):
          f.write(chunk)
      print(f"Image downloaded successfully: {save_path}")
      return save_path
    else:
      print(f"Error downloading image: {response.status_code}")
      return None  # Indicate failure

  except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    return None  # Indicate failure

# Example usage
image_url = "https://www.smartcitiesworld.net/AcuCustom/Sitename/DAM/019/Parsons_PR.jpg"
downloaded_image_path = download_image(image_url)

if downloaded_image_path:
  # You can now use the downloaded image path for further processing
  print(f"Downloaded image path: {downloaded_image_path}")

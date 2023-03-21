import requests

# Make an HTTP GET request to a third-party API that returns the client's IP address
response = requests.get("https://api.ipify.org")

# # Check if the request was successful
# if response.status_code == 200:
#     # Print the client's public IP address
#     print("Your public IP address is:", response.text)
# else:
#     # Handle any errors that occurred
#     print("Error: Request failed with status code", response.status_code)

# Define the API endpoint URL
url = "https://api.mybrowserlocation.com/v1/ip"

# Define any optional parameters, such as the IP address to look up
params = {"ip": response.text}

# Make an HTTP GET request to the API endpoint
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response into a Python dictionary
    location_info = response.json()
    print(location_info)
else:
    # Handle any errors that occurred
    print("Error: Request failed with status code", response.status_code)
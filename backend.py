import base64
import json
from io import BytesIO

import requests
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = ""
s = requests.Session()


def inference(prompt, control, n, modelType):
    if prompt == "" or modelType == "":
        return "Error"

    controlImage64 = ""
    images = []
    with open("output.png", "rb") as image:
        outputImage64 = base64.b64encode(image.read()).decode('utf8')
    if control:
        with open("control.png", "rb") as image:
            controlImage64 = base64.b64encode(image.read()).decode('utf8')
    currentDict = {"image": outputImage64, "prompt": prompt,
                   "controlImage": controlImage64, "modelType": modelType, "mode": "inference"}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    for i in range(n):
        currentDict["it"] = i + 2
        payload = json.dumps(currentDict)
        response = s.post(url, data=payload, headers=headers, verify=False)
        images.append(Image.open(BytesIO(response.content)))
    return images


def training(instancePrompt, validationPrompt, n):
    if validationPrompt == "" or instancePrompt == "":
        return "Error"

    currentDict = {"instancePrompt": instancePrompt, "validationPrompt": validationPrompt, "nOfImages": n,
                   "mode": "training"}

    for i in range(n):
        with open("image" + str(i) + ".png", "rb") as im:
            outputImage64 = base64.b64encode(im.read()).decode('utf8')
            currentDict["image" + str(i)] = outputImage64

    payload = json.dumps(currentDict)
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    response = s.post(url, data=payload, headers=headers, verify=False)

    safetensorsFile = response.content

    return safetensorsFile

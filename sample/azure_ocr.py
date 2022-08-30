# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 08:54:09 2022

@author: user
"""

import json
import time
import requests

url = 'http://10.11.109.75:5100/vision/v3.2/read/analyze?model-version=2022-04-30'

def azure_ocr(image):
    
    # content type
    content_type = 'application/octet-stream'
    headers = {'content-type': content_type}
    
    with open(image, 'rb') as f:
        data = f.read()
    
    # API POST
    response = requests.post(url, headers = headers, data = data)
    
    # Get request ID
    ocr_result_url = response.headers['operation-location']
    
    # Delay
    ocr_result = requests.get(ocr_result_url, time.sleep(1))
    ocr_result_json = json.loads(ocr_result.text)
    
    # Split json result
    result_en = len(ocr_result_json['analyzeResult']['readResults'])
    result_dict = {}
    
    if(result_en):
        print("OCR identifying..")
        ocr_cnt = len(ocr_result_json['analyzeResult']['readResults'][0]['lines'])
        for idx in range(ocr_cnt):
            tmp_result = ocr_result_json['analyzeResult']['readResults'][0]['lines'][idx]['text']
            result_dict[idx] = tmp_result
            
    return result_dict[0]
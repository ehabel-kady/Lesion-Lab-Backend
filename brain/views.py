from django.shortcuts import render
from rest_framework import generics, mixins
from rest_framework import permissions
from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.response import Response
from django.utils.translation import gettext as _
import math, os, pydicom
from brain.models import *
from brain.serializers import *
from django.utils import timezone , dateparse
from zipfile import *
from django.conf import settings
from io import BytesIO
from urllib.request import urlopen
import pydicom
from PIL import Image
import shutil
import cv2
import numpy as np
from dl.dln import *
# Create your views here.

class CreatePatient(generics.CreateAPIView):
    serializer_class = PatientSerializer
    permission_class = [permissions.AllowAny]
    message_sucess = "Patient created successfully"

    def perform_create(self, request):
        """perform create method that returns a scan instance"""
        id = request.data['patient_id']
        name = request.data['name']
        age = request.data['age']
        weight = request.data['weight']
        gender = request.data['gender']
        return Patient.objects.get_or_create(patient_id=id, name=name, age=age, gender=gender, weight=weight)
    def create(self, request, *args, **kwargs):
        state= self.perform_create(request)
        if state[1]:
            return Response(self.message_sucess,status=status.HTTP_201_CREATED)    
        return Response("patient created before",status=status.HTTP_201_CREATED)


class CreateScans(generics.CreateAPIView):
    serializer_class = ScanSerializer
    permission_class = [permissions.AllowAny]

    def perform_create(self, request):
        """perform create method that returns a scan instance"""
        return "created"
    def create(self, request, *args, **kwargs):
        # print(request.data)
        print(request.data['setOne'][0])
        
        
        return Response("patient created before",status=status.HTTP_201_CREATED)

def dicomSort(path_to_scans, file_array):
        dicom_sorting = {}
        for file in range(len(file_array)):
            instance = pydicom.dcmread(path_to_scans + "/" + file_array[file])
            instance_value = instance.InstanceNumber
            dicom_sorting[instance_value] = file_array[file]
        sorted(dicom_sorting.keys())
        return dicom_sorting


class CreateSets(generics.CreateAPIView):
    serializer_class = SetSerializer
    permission_class = [permissions.AllowAny]
    
    """
    IMPORTANT:
    NEED TO ADD A SNIPPET TO RESIZE THE IMAGES INPUT TO 256x256
    """
    """
        USE CASE
        - The data payload contains the data from zip file uploaded by
        the doctor.
        - Navigate to uploads folder /BASE_DIR/uploads/ (Could be improved by dynamically creating it too.)
        - Directory for patient created and named after "name" in request payload.
        - Temp ZIP file created for extraction process
        - Files extracted
        - ZIP file deleted
    """
    def create(self, request, *args, **kwargs):
        os.chdir(settings.BASE_DIR)
        media_dir = settings.BASE_DIR + '/uploads/'
        os.chdir(media_dir)
        if (request.data[0]["name"].split('.zip')[0]) in os.listdir(media_dir):
            shutil.rmtree(request.data[0]["name"].split('.zip')[0])
        patient_media = media_dir+request.data[0]["name"].split('.zip')[0]
        os.mkdir(patient_media)
        os.chdir(patient_media)
        print("Navigated to patient media directory")
        zip_data = request.data[0]["data"]
        
        url_zip = urlopen(zip_data)
        fileNaming = media_dir+"temp.zip"
        temp_zip = open(fileNaming, "wb")
        temp_zip.write(url_zip.read())
        temp_zip.close()
        zf = ZipFile(fileNaming)
        zf.extractall()
        zf.close()
        os.chdir(media_dir)
        os.remove("temp.zip")
        print("Patient Data Uploaded")

        # Navigate to Patient Directory and create images subfolder
        os.chdir(patient_media)
        os.mkdir('images')
        images_dir = patient_media+'/images/old/'
        predictions_dir = patient_media+'/images/new/'
        os.mkdir(images_dir)
        os.mkdir(predictions_dir)
        os.chdir(images_dir)

        patient_media_dirs = os.listdir(patient_media)

        model = UNet()
        path_to_weights = settings.BASE_DIR + '/dl/weights.h5'
        model.load_weights(path_to_weights)

        # Sorting dicoms by instance numbers
        for scan_set in patient_media_dirs:
            if scan_set == 'images':
                pass
            else:
                # Generating the original scans
                current_set_path = patient_media+'/'+scan_set
                sorted_map = dicomSort(current_set_path, os.listdir(current_set_path))
                # print(sorted_map)
                # print('======================')
                indexer = 1
                for image in sorted(sorted_map.keys()):
                    # print(sorted_map[image])
                    # print(current_set_path)
                    set_dir_path = images_dir+scan_set
                    if os.path.exists(set_dir_path):
                        pass
                    else:
                        os.mkdir(set_dir_path)
                    os.chdir(set_dir_path)
                    ds = pydicom.dcmread(current_set_path+"/"+str(sorted_map[image])).pixel_array
                    if ds.shape != (256,256):
                        ds = cv2.resize(ds, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
                    else:
                        pass
                    naming = str(indexer)+'.png'
                    cv2.imwrite(naming, ds)
                    indexer+=1
                
                # Generating the model predictions
                current_set_path = patient_media+'/'+scan_set
                sorted_map = dicomSort(current_set_path, os.listdir(current_set_path))
                indexer = 1
                
                for image in sorted(sorted_map.keys()):
                    set_images = []
                    set_dir_path = predictions_dir+scan_set
                    if os.path.exists(set_dir_path):
                        pass
                    else:
                        os.mkdir(set_dir_path)
                    os.chdir(set_dir_path)
                    ds = pydicom.dcmread(current_set_path+"/"+str(sorted_map[image])).pixel_array
                    if ds.shape != (256,256):
                        ds = cv2.resize(ds, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
                    else:
                        pass
                    set_images.append(ds)
                    generatePredictions(model, set_images, indexer)
                    indexer+=1


        



        """
        TLDR;
        
        - Operate on the files as follows:
            1. Get the patient gender, ID and other metadata from only one dcm file
            2. Retreived data is later on sent to the platform as props
            3. Sort the files based on instanceNumber using dicomSort routine
            4. Instantiate the patient model with the following
                {
                    patient_id
                    gender
                    age
                    weight (if available)
                    setOne [ [original] [predictions] ]
                    setTwo [ [original] [predictions] ]
                    setThree [ [original] [predictions] ]
                    setFour [ [original] [predictions] ]
                    
                }
        
        """

        serializer = self.get_serializer(data=request.data,many=True)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


from django.shortcuts import render
from rest_framework import generics, mixins
from rest_framework import permissions
from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.response import Response
from django.utils.translation import gettext as _
import math, os, pydicom
from brain.models import *
from django.http import HttpResponse
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
import shutil
import json
from dl.dln import *

# Create your views here.

class JSONResponse(HttpResponse):
    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content,**kwargs)

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

    def extrct_data(self, request):
        os.chdir(settings.BASE_DIR)
        self.media_dir = settings.BASE_DIR + '/uploads/'
        os.chdir(self.media_dir)
        self.patient_media = self.media_dir+request.data[0]["name"].split('.zip')[0]
        os.mkdir(self.patient_media)
        os.chdir(self.patient_media)
        self.zip_data = request.data[0]["data"]
        self.url_zip = urlopen(self.zip_data)
        self.fileNaming = self.media_dir+"temp.zip"
        self.temp_zip = open(self.fileNaming, "wb")
        self.temp_zip.write(self.url_zip.read())
        self.temp_zip.close()
        self.zf = ZipFile(self.fileNaming)
        self.zf.extractall()
        self.zf.close()
        os.chdir(self.media_dir)
        os.remove("temp.zip")

    def create(self, request, *args, **kwargs):
        self.extrct_data(request)
        print("MEDIA_DIR: ", self.patient_media)
        print("Patient Data Uploaded")
        model = UNet()
        path_to_weights = settings.BASE_DIR + '/dl/weights.h5'
        model.load_weights(path_to_weights)
        # Navigate to Patient Directory and create images subfolder
        os.chdir(self.patient_media)
        scans = dict()
        for r, d, f in os.walk(self.patient_media+'/'):
            for img in f:
                instance=pydicom.dcmread(r+'/'+img)
                scan_type=instance[0x0008, 0x103e].value
                patient_name=instance[0x0010,0x0010].value
                patient_id=instance[0x0010,0x0020].value
                birth_date=instance[0x0010,0x0030].value
                sex = instance[0x0010,0x0040].value
                age=int(instance[0x0010,0x1010].value.split('Y')[0])
                weight=instance[0x0010,0x1030].value
                instance_number=instance.InstanceNumber
                new_path = settings.MEDIA_ROOT+'/scans/'+scan_type+'/'
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                if not scan_type in scans:
                    scans[scan_type] = []
                pixel_array_numpy=instance.pixel_array
                if pixel_array_numpy.shape != (256,256):
                    pixel_array_numpy = cv2.resize(pixel_array_numpy, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
                os.rename(r+'/'+img, new_path+img)
                img=img.replace('.dcm', '.png')
                # pixel_array_numpy=instance.pixel_array
                cv2.imwrite(new_path+img, pixel_array_numpy)
                patient, created = Patient.objects.get_or_create(patient_id=patient_id, name=patient_name, gender=sex, age=age)
                type_scan, created= Set.objects.get_or_create(name=scan_type)
                Scan.objects.create(scan_image=new_path+img, instance_number=instance_number, sets=type_scan, patient=patient, stage='old')
        # Sorting dicoms by instance 
        set_images=[]
        for scan in scans:
            scans[scan] = Scan.objects.filter(sets__name=scan, patient=patient).values('scan_image')
        scan_images = dict()
        for scan in scans:
            indexer=1
            scan_images[scan]=[]
            for image in scans[scan]:
                if '.png' in image['scan_image']:
                    scan_images[scan].append(image['scan_image'])
                else:
                    ds = pydicom.dcmread(current_set_path+"/"+str(sorted_map[image])).pixel_array
                    set_images.append(ds)
                    generatePredictions(model, set_images, indexer)
                    indexer+=1
        response_data = {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "gender": sex,
            "age": age,
            "scan_images": scan_images
        }
        # print(response_data)
        shutil.rmtree(self.patient_media)
<<<<<<< HEAD
<<<<<<< HEAD
        serializer = self.get_serializer(data=scans)
        serializer.is_valid()
        headers = self.get_success_headers(serializer.data)
        print(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, content_type = 'application/javascript; charset=utf8')
=======
        # return JSONResponse(json.dumps(response_data), status=status.HTTP_201_CREATED)
        return Response("Done", status=status.HTTP_201_CREATED)
>>>>>>> a5a01eee5d695d9165fe8019387bb1ae8d556629
=======
        # return JSONResponse(json.dumps(response_data), status=status.HTTP_201_CREATED)
        return Response("Done", status=status.HTTP_201_CREATED)
>>>>>>> a5a01eee5d695d9165fe8019387bb1ae8d556629

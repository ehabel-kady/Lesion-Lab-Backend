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

class CreateSets(generics.CreateAPIView):
    serializer_class = SetSerializer
    permission_class = [permissions.AllowAny]
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

        serializer = self.get_serializer(data=request.data,many=True)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


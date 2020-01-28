from django.shortcuts import render
from rest_framework import generics, mixins
from rest_framework import permissions
from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.response import Response
from django.utils.translation import gettext as _
import math, os, pydicom
from brain.models import *
from django.http import HttpResponse, JsonResponse
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
    # print(path_to_scans)
    dicom_sorting = {}
    for fileName in range(len(file_array)):
        # print(pydicom.dcmread(path_to_scans + file_array[filex], force=True))
        instance = pydicom.dcmread(path_to_scans + file_array[fileName])
        instance_value = instance.InstanceNumber
        dicom_sorting[instance_value] = file_array[fileName]
    sorted(dicom_sorting.keys())
    return dicom_sorting


class CreateSets(generics.CreateAPIView):
    serializer_class = SetSerializer
    permission_class = [permissions.AllowAny]
    tf.reset_default_graph()

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
        Set.objects.all().delete()
        Scan.objects.all().delete()
        Patient.objects.all().delete()
        self.extrct_data(request)
        print("MEDIA_DIR: ", self.patient_media)
        print("Patient Data Uploaded")
        # Compiling model and loading weights
        model = UNet()
        path_to_weights = settings.BASE_DIR + '/dl/weights.h5'
        model.load_weights(path_to_weights)
        # Navigate to Patient Directory and create images subfolder
        os.chdir(self.patient_media)
        scans = dict()
        scan_types = []
        for r, d, f in os.walk(self.patient_media+'/'):
            for img in f:
                instance=pydicom.dcmread(r+'/'+img)
                scan_type=instance[0x0008, 0x103e].value
                patient_name=request.data[0]["name"].split('.zip')[0]
                patient_id=instance[0x0010,0x0020].value
                birth_date=instance[0x0010,0x0030].value
                sex = instance[0x0010,0x0040].value
                age=instance[0x0010,0x1010].value
                # First change here after benchmark
                sop_uid=str(instance[0x0008, 0x0018].value)
                study_date=str(instance[0x0008, 0x0020].value)
                slice_spacing=str(instance[0x0018, 0x0088].value)
                manufacturer=str(instance.Manufacturer)
                pixel_spacing=str(instance[0x0028, 0x0030].value)
                organ=str(instance[0x0018, 0x0015].value)
                slice_thickness=str(instance[0x0018, 0x0050].value)
                mfs=str(instance[0x0018, 0x0087].value) # Magnetic Field Strength
                observer_name=str(instance[0x0040, 0xa075].value)
                p_name=str(instance[0x0040, 0xa123].value)
                # End of edits
                weight=instance[0x0010,0x1030].value
                instance_number=instance.InstanceNumber
                original_path = settings.MEDIA_ROOT+'/scans/old/'+scan_type+'/'
                prediction_path = settings.MEDIA_ROOT+'/scans/new/'+scan_type+'/'
                os.makedirs(original_path, exist_ok=True)
                os.makedirs(prediction_path, exist_ok=True)
                if not scan_type in scans:
                    scans[scan_type] = []
                    scan_types.append(scan_type)
                pixel_array_numpy=instance.pixel_array
                if pixel_array_numpy.shape != (256,256):
                    pixel_array_numpy = cv2.resize(pixel_array_numpy, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
                os.rename(r+'/'+img, original_path+img)
                img=img.replace('.dcm', '.png')
                # pixel_array_numpy=instance.pixel_array
                cv2.imwrite(original_path+img, pixel_array_numpy)
                patient, created = Patient.objects.get_or_create(patient_id=patient_id, name=patient_name, gender=sex, age=age)
                type_scan, created= Set.objects.get_or_create(name=scan_type)
                Scan.objects.create(scan_image=original_path+img, instance_number=instance_number, sets=type_scan, patient=patient, stage='old')
        # Sorting dicoms by instance 
        set_images=[]
        for scan in scans:
            scans[scan] = Scan.objects.filter(sets__name=scan, patient=patient, stage="old").order_by('instance_number').values('scan_image')
        scan_images = dict()
        list_of_files = dict()
        
        for scan in scans:
            dcm_scans=os.listdir(settings.MEDIA_ROOT+'/scans/old/'+scan+'/')
            list_of_files[scan] = []
            for dcmFile in dcm_scans:
                if '.png' in dcmFile:
                    pass
                else:
                    # list_of_files[scan].append('2')
                    list_of_files[scan].append(settings.MEDIA_ROOT+'/scans/old/'+scan+'/'+dcmFile)
            scan_images[scan]=[]
            for image in scans[scan]:
                if '.png' in image['scan_image']:
                    scan_images[scan].append(image['scan_image'])
        scan_images_new = dict()
        # Modifying in predictions start here
        # ----------------------------------------------------------
        sorted_file_lists = dict()
        for scan_set in scans:
            path_to_set = settings.MEDIA_ROOT+'/scans/old/'+scan_set+'/'
            set_files = os.listdir(path_to_set)
            set_files_final = []
            for filedata in set_files:
                if '.dcm' in filedata:
                    set_files_final.append(filedata)
                else:
                    pass
            
            sorted_file_lists[scan_set] = dicomSort(path_to_set, set_files_final)
        
        path_array = []
        path_array_old = []
        types_scans = []
        for paths in sorted_file_lists:
            types_scans.append(paths)
            path_array.append(settings.MEDIA_ROOT+'/scans/new/'+paths+'/')
            path_array_old.append(settings.MEDIA_ROOT+'/scans/old/'+paths+'/')
        
        # iterator = list(sorted_file_lists.keys())[0]
        keys_sorted = sorted(sorted_file_lists[types_scans[0]])
        # print('------------------------')
        # print(path_array[0])
        # print('------------------------')
        # sorted_file_lists[types_scans[0]][3]
        # print('------------------------')
        # # x = sorted(sorted_file_lists['AX FLAIR'])
        
        # print(path_array[0]+sorted_file_lists[types_scans[0]][3])
        # print('------------------------')
        # print('------------------------')
        # print('------------------------')


        for imgKey in range(len(keys_sorted)):
            dcmArr = []

            im_one = pydicom.dcmread(path_array_old[0]+sorted_file_lists[types_scans[0]][keys_sorted[imgKey]]).pixel_array
            im_two = pydicom.dcmread(path_array_old[1]+sorted_file_lists[types_scans[1]][keys_sorted[imgKey]]).pixel_array
            im_three = pydicom.dcmread(path_array_old[2]+sorted_file_lists[types_scans[2]][keys_sorted[imgKey]]).pixel_array
            im_four = pydicom.dcmread(path_array_old[3]+sorted_file_lists[types_scans[3]][keys_sorted[imgKey]]).pixel_array
            
            
            dcmArr=[im_one, im_two, im_three, im_four]
            generatePredictions(model, dcmArr, path_array, imgKey)
            
            type_one = type_scan=Set.objects.filter(name=types_scans[0])[0]
            type_two = type_scan=Set.objects.filter(name=types_scans[1])[0]
            type_three = type_scan=Set.objects.filter(name=types_scans[2])[0]
            type_four = type_scan=Set.objects.filter(name=types_scans[3])[0]
            
            Scan.objects.create(scan_image=path_array[0]+str(imgKey)+'_original.png', instance_number=imgKey, sets=type_one, patient=patient, stage='new')
            Scan.objects.create(scan_image=path_array[1]+str(imgKey)+'_original.png', instance_number=imgKey, sets=type_two, patient=patient, stage='new')
            Scan.objects.create(scan_image=path_array[2]+str(imgKey)+'_original.png', instance_number=imgKey, sets=type_three, patient=patient, stage='new')
            Scan.objects.create(scan_image=path_array[3]+str(imgKey)+'_original.png', instance_number=imgKey, sets=type_four, patient=patient, stage='new')

        # ----------------------------------------------------------
        # for scan_set in list_of_files:
        #     os.chdir(settings.MEDIA_ROOT+'/scans/new/'+scan_set+'/')
        #     indexer=1
        #     type_scan=Set.objects.filter(name=scan_set)[0]
        #     # scan_images_new[scan_set] = []
        #     for scan_slice in list_of_files[scan_set]:
        #         temp = []
        #         ds = scan_slice.pixel_array
        #         temp.append(ds)
        #         instance_number=scan_slice.InstanceNumber
        #         generatePredictions(model, temp, scan_slice.InstanceNumber)
        #         prediction_path = settings.MEDIA_ROOT+'/scans/new/'+scan_set+'/'+str(instance_number)+'_original.png'
        #         # scan_images_new[scan_set].append(prediction_path)
        #         Scan.objects.create(scan_image=prediction_path, instance_number=instance_number, sets=type_scan, patient=patient, stage='new')
        scan_images_returned = dict()
        for scan in scans:
            scan_images_returned[scan] = []
            scan_images_new[scan] = Scan.objects.filter(sets__name=scan, patient=patient, stage="new").order_by('instance_number').values('scan_image')
            for image in scan_images_new[scan]:
                if '.png' in image['scan_image']:
                    scan_images_returned[scan].append(image['scan_image'])

        print(scans)
        response_data = {
            # "patient_name": patient_name,
            "patient_id": patient_id,
            "gender": sex,
            "age": age,
            "scan_images_old": scan_images,
            "scan_images_new": scan_images_returned,
            "type_0":scan_types[0],
            "type_1":scan_types[1],
            "type_2":scan_types[2],
            "type_3":scan_types[3],
            "sop_uid":sop_uid,
            "date":study_date,
            "weight":weight,
            "slice_spacng":slice_spacing,
            "manufacturer":manufacturer,
            "pixel_spacing":pixel_spacing,
            "organ":organ,
            "thickness":slice_thickness,
            "mf_strength":mfs,
            "observer":observer_name,
            "subject":p_name

        }
        print(response_data)
        shutil.rmtree(self.patient_media)
        serializer = self.get_serializer(data=scans)
        serializer.is_valid()
        headers = self.get_success_headers(serializer.data)
        print(serializer.data)
        return JsonResponse(json.dumps(response_data), status=status.HTTP_201_CREATED, content_type = 'application/javascript; charset=utf8', safe=False)

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
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data,many=True)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


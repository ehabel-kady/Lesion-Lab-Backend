from django.shortcuts import render
from rest_framework import generics, mixins
from rest_framework import permissions
from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.response import Response
from django.utils.translation import gettext as _
import math
from brain.models import *
from brain.serializers import *
from django.utils import timezone , dateparse
# Create your views here.

class CreateScans(generics.CreateAPIView):
    serializer_class = ScanSerializer
    permission_class = [permissions.AllowAny]

    def perform_create(self, serializer):
        """perform create method that returns a scan instance"""
        return serializer.save()
    
    
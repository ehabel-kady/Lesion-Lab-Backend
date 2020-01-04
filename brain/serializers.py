from rest_framework import serializers
from brain.models import *

class PatientSerializer(serializers.Serializer):
    def create(self, validated_data):
        """Create and return new patint instance"""
        return Patient.objects.create(**validated_data)
    class Meta:
        model = Patient
        fields ='__all__'

class ScanSerializer(serializers.Serializer):
    def create(self, validated_data):
        """Create and return new scan instance"""
        return Scan.objects.create(**validated_data)
    class Meta:
        model = Scan
        fields ='__all__'
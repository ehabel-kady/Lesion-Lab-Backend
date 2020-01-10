from rest_framework import serializers
from brain.models import *

class PatientSerializer(serializers.ModelSerializer):
    
    def create(self, validated_data):
        """Create and return new patint instance"""
        return Patient.objects.create(**validated_data)
    class Meta:
        model = Patient
        fields ='__all__'

class ScanSerializer(serializers.ModelSerializer):
    
    def create(self, validated_data):
        """Create and return new scan instance"""
        return Scan.objects.create(**validated_data)
    class Meta:
        model = Scan
        fields =('scan_image',)


class SetSerializer(serializers.Serializer):
    scans = ScanSerializer(many=True)
    name = serializers.CharField(max_length=20)
    def create(self, validated_data):
        scans = validated_data.pop('scans')
        se = Set.objects.create(**validated_data)
        for i in scans:
            Scan.objects.create(**i,sets=se)
        """Create and return new scan instance"""
        return se
    class Meta:
        model = Set
        fields =('name','scans',)

# class SetGroupSerializer(serializers.Serializer):
#     sets = SetSerializer(many=True)
#     def create(self, validated_data):
#         sets = validated_data.pop('sets')
#         ser = SetSerializer(data=sets)
#         ser.is_valid(raise_exception=True)
#         ser.save()
#         se = Set.objects.create(**validated_data)
#         for i in scans:
#             Scan.objects.create(**i,sets=se)
#         """Create and return new scan instance"""
#         return se
#     class Meta:
#         model = SetGroup
#         fields =('sets',)
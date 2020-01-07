from django.db import models

# Create your models here.
gender_choices = [
    ('male', 'M'),
    ('Female', 'F'),
    ('None', 'N')
]
class Patient(models.Model):
    """Model Definition for patient"""
    patient_id=models.CharField(max_length=50, unique=True)
    name=models.CharField(max_length=150, null=True, blank=True)
    age=models.PositiveIntegerField()
    weight=models.DecimalField(max_digits=7, decimal_places=2)
    gender=models.CharField(choices=gender_choices,default='None',max_length=10)

    class Meta:
        """Meta definition for Patient."""

        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'

    def __str__(self):
        """Unicode representation of Patient."""
        return (self.patient_id)

class Scan(models.Model):
    scan_image = models.FileField(upload_to="scans/", null=True, blank=True)
    instance_number = models.PositiveIntegerField()
    scan_type = models.CharField(max_length=50)
    patient = models.ForeignKey('brain.Patient', related_name='patient', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        """Meta definition for Scan."""

        verbose_name = 'Scan'
        verbose_name_plural = 'Scans'

    def __str__(self):
        """Unicode representation of Scan."""
        return (self.id)

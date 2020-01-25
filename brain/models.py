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
    age=models.PositiveIntegerField(null=True)
    weight=models.FloatField(null=True)
    gender=models.CharField(choices=gender_choices,default='None',max_length=10, null=True)

    class Meta:
        """Meta definition for Patient."""

        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'

    def __str__(self):
        """Unicode representation of Patient."""
        return (self.patient_id)

class Scan(models.Model):
    scan_image = models.FileField(upload_to="scans/", null=True, blank=True)
    instance_number = models.PositiveIntegerField(null=True, blank=True)
    scan_type = models.CharField(max_length=50,null=True, blank=True)
    stage = models.CharField(max_length=10, null=True, blank=True)
    patient = models.ForeignKey('brain.Patient', related_name='patient', on_delete=models.CASCADE, null=True, blank=True)
    sets  = models.ForeignKey('brain.Set', related_name='scans', on_delete=models.CASCADE,null=True,blank=True)
    class Meta:
        """Meta definition for Scan."""

        verbose_name = 'Scan'
        verbose_name_plural = 'Scans'

    def __str__(self):
        """Unicode representation of Scan."""
        return str(self.id)

class Set(models.Model):
    """Model definition for Set."""
    # scan = models.ForeignKey('brain.Scan', related_name='scans', on_delete=models.CASCADE, null=True, blank=True)
    # TODO: Define fields here
    name = models.CharField(max_length=20, blank=True, null=True)
    class Meta:
        """Meta definition for Set."""

        verbose_name = 'Set'
        verbose_name_plural = 'Sets'

    def __str__(self):
        """Unicode representation of Set."""
        return self.name
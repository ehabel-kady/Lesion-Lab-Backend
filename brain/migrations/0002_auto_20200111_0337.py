# Generated by Django 2.2.6 on 2020-01-11 03:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('brain', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='scan',
            name='sets',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='scans', to='brain.Set'),
        ),
    ]

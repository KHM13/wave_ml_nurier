# Generated by Django 3.2.16 on 2023-09-19 15:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('preprocess', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mldatasetfile',
            name='dataset_file',
            field=models.FileField(default='', unique=True, upload_to='file/dataset/%Y%m%d'),
        ),
        migrations.AlterField(
            model_name='mldatasetfile',
            name='dataset_file_extension',
            field=models.TextField(default='csv'),
        ),
    ]

# Generated by Django 3.2.16 on 2023-09-19 15:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('preprocess', '0002_auto_20230919_1506'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mldatasetfile',
            name='dataset_file',
            field=models.TextField(unique=True),
        ),
    ]

# Generated by Django 2.2.6 on 2022-07-04 02:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0005_presentation'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadmnist',
            name='mnistfile',
            field=models.FileField(upload_to='media/mnist'),
        ),
        migrations.AlterField(
            model_name='uploadmodel',
            name='modelfile',
            field=models.FileField(upload_to='media/models'),
        ),
    ]

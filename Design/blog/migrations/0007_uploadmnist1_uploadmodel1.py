# Generated by Django 2.2.6 on 2022-07-04 02:51

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0006_auto_20220704_1016'),
    ]

    operations = [
        migrations.CreateModel(
            name='uploadmnist1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=10)),
                ('mnist', models.FileField(upload_to='mnist')),
                ('created_time', models.CharField(default=django.utils.timezone.now, max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Uploadmodel1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=10)),
                ('modelfile', models.FileField(upload_to='models')),
                ('created_time', models.CharField(default=django.utils.timezone.now, max_length=50)),
            ],
        ),
    ]
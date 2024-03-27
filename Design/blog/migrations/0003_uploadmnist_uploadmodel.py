# Generated by Django 2.2.6 on 2022-06-21 06:13

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0002_article_articlecomment_category_tag'),
    ]

    operations = [
        migrations.CreateModel(
            name='uploadmnist',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mnistname', models.CharField(max_length=20)),
                ('mnistfile', models.FileField(upload_to='mnistandmodel')),
                ('created_time', models.CharField(default=django.utils.timezone.now, max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='uploadmodel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('modelname', models.CharField(max_length=20)),
                ('modelfile', models.FileField(upload_to='mnistandmodel')),
                ('created_time', models.CharField(default=django.utils.timezone.now, max_length=50)),
            ],
        ),
    ]

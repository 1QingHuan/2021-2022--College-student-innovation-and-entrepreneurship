# Generated by Django 2.2.6 on 2022-06-21 11:36

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0004_auto_20220621_1557'),
    ]

    operations = [
        migrations.CreateModel(
            name='presentation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('kmnc', models.FloatField(max_length=20)),
                ('nbc', models.FloatField(max_length=20)),
                ('snac', models.FloatField(max_length=20)),
                ('tknc', models.FloatField(max_length=20)),
                ('created_time', models.CharField(default=django.utils.timezone.now, max_length=50)),
            ],
            options={
                'ordering': ['created_time'],
            },
        ),
    ]

# Generated by Django 4.0.3 on 2022-05-12 11:29

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0010_survey_adhar'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='survey',
            name='user',
        ),
    ]

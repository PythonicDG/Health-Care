# Generated by Django 4.1.7 on 2023-04-09 20:39

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('home', '0017_alter_survey_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='survey',
            name='user',
            field=models.OneToOneField(blank=True, max_length=20, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]

# Generated by Django 4.1.7 on 2023-04-09 19:17

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('home', '0013_contact_us'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Contact_us',
        ),
        migrations.AddField(
            model_name='survey',
            name='user',
            field=models.OneToOneField(default='', max_length=20, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
    ]

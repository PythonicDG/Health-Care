# Generated by Django 4.0.3 on 2022-05-12 10:17

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('home', '0008_customuser_issue_date_customuser_issue_no_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Survey',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('no_of_family_members', models.CharField(blank=True, max_length=20, null=True)),
                ('name_of_head_of_family', models.CharField(blank=True, max_length=20, null=True)),
                ('age_of_head_of_family', models.CharField(blank=True, max_length=20, null=True)),
                ('no_of_other_adult_males', models.CharField(blank=True, max_length=20, null=True)),
                ('no_of_other_adult_female', models.CharField(blank=True, max_length=20, null=True)),
                ('no_of_male_children', models.CharField(blank=True, max_length=20, null=True)),
                ('no_of_female_children', models.CharField(blank=True, max_length=20, null=True)),
                ('water_source', models.CharField(blank=True, max_length=20, null=True)),
                ('water_distance', models.CharField(blank=True, max_length=20, null=True)),
                ('water_treatment', models.CharField(blank=True, max_length=20, null=True)),
                ('non_usage_of_toilets', models.CharField(blank=True, max_length=20, null=True)),
                ('distance_of_subcentres', models.CharField(blank=True, max_length=20, null=True)),
                ('distance_of_primary_centers', models.CharField(blank=True, max_length=20, null=True)),
                ('distance_of_community_centers', models.CharField(blank=True, max_length=20, null=True)),
                ('distance_of_district_hospital', models.CharField(blank=True, max_length=20, null=True)),
                ('distance_of_patahology_lab', models.CharField(blank=True, max_length=20, null=True)),
                ('distance_of_medical_store', models.CharField(blank=True, max_length=20, null=True)),
                ('status_of_delivery_of_children', models.CharField(blank=True, max_length=20, null=True)),
                ('status_of_vaccination_of_children', models.CharField(blank=True, max_length=20, null=True)),
                ('status_of_female_problems', models.CharField(blank=True, max_length=20, null=True)),
                ('centrally_issued_health_insurance', models.CharField(blank=True, max_length=20, null=True)),
                ('state_issued_health_insurance', models.CharField(blank=True, max_length=20, null=True)),
                ('personal_health_insurance', models.CharField(blank=True, max_length=20, null=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]

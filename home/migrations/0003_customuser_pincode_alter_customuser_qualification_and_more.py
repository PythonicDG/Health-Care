# Generated by Django 4.0.3 on 2022-05-09 11:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_customuser_symptoms'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='pincode',
            field=models.CharField(default='123456', max_length=6),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='qualification',
            field=models.CharField(blank=True, choices=[('No Formal education', 'No Formal education'), ('Primary Education', 'Primary Educations'), ('Secondary Education', 'Secondary Education'), ('Post Graduation diploma', 'Post Graduation Diploma'), ('Vocational Qualification', 'Vocational Qualification'), ('Bachelors Degree', 'Bachelors Degree'), ('Masters Degree', 'Masters Degree'), ('Doctorate or Higher', 'Doctorate or Higher')], default=None, max_length=30, null=True),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='specialization',
            field=models.CharField(blank=True, choices=[('Doctor', 'Doctor'), ('Sevika', 'Sevika'), ('NGO', 'NGA')], max_length=30, null=True),
        ),
    ]

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
# from django.utils.translation import gettext_lazy as _

# Create your models here.
specialization=(('Doctor','Doctor'),('Screener','Screener'),('Sevika','Sevika'),('Ngo','Ngo'),('Pharmacy','Pharmacy'))


class CustomUser(models.Model):
    
    user=models.OneToOneField(User,on_delete=models.CASCADE)
    upload = models.ImageField(upload_to='static/image')
    mobile=models.BigIntegerField(null=True)
    specialization = models.CharField(max_length=30, choices=specialization,blank=True,null=True)
    gender = models.CharField(max_length=6, choices=[('Male','Male'),('Female','Female')])
    qualification= models.CharField(max_length=30, choices=[('No Formal education','No Formal education'),('Primary Education','Primary Educations'),('Secondary Education','Secondary Education'),('Post Graduation diploma','Post Graduation Diploma'),('Vocational Qualification','Vocational Qualification'),('Bachelors Degree','Bachelors Degree'),('Masters Degree','Masters Degree'),('Doctorate or Higher','Doctorate or Higher')],default=None,null=True,blank=True)
    dob=models.DateField()
    blood_grp = models.CharField(max_length=20, choices=[('A+','A+'),('A-','A-'),('B+','B+'),('B-','B-'),('O+','O+'),('O-','O-'),('AB+','AB+'),('AB-','AB-')])
    country=models.CharField(default='India',max_length=10)
    state=models.CharField(max_length=30)
    district=models.CharField(max_length=30)
    address=models.CharField(max_length=100)
    aadhar = models.CharField(max_length=12,unique=True)
    pincode= models.CharField(max_length=6,default='123456')
    
    
    
   



    symptoms = models.CharField(max_length=1000,null=True,blank=True)
    disease=models.CharField(max_length=255,null=True,blank=True)
    height = models.CharField(max_length=10,null=True,blank=True)
    weight = models.CharField(max_length=10,null=True,blank=True)
    bp_sys= models.CharField(max_length=10,null=True,blank=True)
    bp_dia=models.CharField(max_length=10,null=True,blank=True)
    spo2=models.CharField(max_length=10,null=True,blank=True)
    pulse=models.CharField(max_length=10,null=True,blank=True)
    temp=models.CharField(max_length=10,null=True,blank=True)
    arm=models.CharField(max_length=10,null=True,blank=True)
    Issue_no=models.CharField(max_length=20,null=True,blank=True)
    Issue_date=models.CharField(max_length=20,null=True,blank=True)
    subject = models.CharField(max_length=2002,null=True,blank=True)
    issue= models.CharField(max_length=20,null=True,blank=True)




class Survey(models.Model):
    user=models.OneToOneField(User,on_delete=models.CASCADE,max_length=20,null=True,default=None)
    adhar=models.CharField(max_length=12,null=True,blank=True,unique=True)
    no_of_family_members=models.CharField(max_length=20,null=True,blank=True)
    name_of_head_of_family=models.CharField(max_length=20,null=True,blank=True)
    age_of_head_of_family=models.CharField(max_length=20,null=True,blank=True)
    no_of_other_adult_males=models.CharField(max_length=20,null=True,blank=True)
    no_of_other_adult_female=models.CharField(max_length=20,null=True,blank=True)
    no_of_male_children=models.CharField(max_length=20,null=True,blank=True)
    no_of_female_children=models.CharField(max_length=20,null=True,blank=True)
    water_source=models.CharField(max_length=20,null=True,blank=True)
    water_distance=models.CharField(max_length=20,null=True,blank=True)
    water_treatment=models.CharField(max_length=20,null=True,blank=True)
    non_usage_of_toilets=models.CharField(max_length=20,null=True,blank=True)
    distance_of_subcentres=models.CharField(max_length=20,null=True,blank=True)
    distance_of_primary_centers=models.CharField(max_length=20,null=True,blank=True)
    distance_of_community_centers=models.CharField(max_length=20,null=True,blank=True)
    distance_of_district_hospital=models.CharField(max_length=20,null=True,blank=True)
    distance_of_patahology_lab=models.CharField(max_length=20,null=True,blank=True)
    distance_of_medical_store=models.CharField(max_length=20,null=True,blank=True)
    status_of_delivery_of_children=models.CharField(max_length=20,null=True,blank=True)
    status_of_vaccination_of_children=models.CharField(max_length=20,null=True,blank=True)
    status_of_female_problems=models.CharField(max_length=20,null=True,blank=True)
    centrally_issued_health_insurance=models.CharField(max_length=20,null=True,blank=True)
    state_issued_health_insurance=models.CharField(max_length=20,null=True,blank=True)
    personal_health_insurance=models.CharField(max_length=20,null=True,blank=True)

    

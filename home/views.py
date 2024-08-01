import random
from django.shortcuts import render, redirect
from django.contrib.auth.models import auth, User
from django.urls import reverse
from django.core.mail import send_mail, BadHeaderError
from django.http import HttpResponse
from django.contrib.auth.forms import PasswordResetForm
from django.template.loader import render_to_string
from django.db.models.query_utils import Q
from django.utils.http import urlsafe_base64_encode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes
from django.contrib import messages
from home.models import CustomUser, Survey
import smtplib
from django.contrib import messages
from django.core.mail import send_mail



# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import date


today=date.today()
d1=today.strftime("%A, %B %d, %Y")


# Create your views here.
def index(request):
    return render(request,'index.html')

def dashboard(request):
	return render(request,'admin_dashboard.html')


def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('dipakgaikwadmg@gmail.com', 'uhbfgvgbyvmadnvk')
    server.sendmail('dipakgaikwadmg@gmail.com', 'dipakgaikwadmg@gmail.com')
    server.close()

def user_login(request):
    
    if request.method=='POST':
        username=request.POST['username']
        password=request.POST['password']
        user = auth.authenticate(username=username,password=password)
		

        if user is not None:
            auth.login(request,user)
	    	
	    	
	    
	    	
            return redirect(reverse('home:dashboard'))
		
			
	
	

        else:
            print('admin login error')
            return redirect(reverse('home:user_login'))
    else:
        return render(request, 'user_login.html')

def password_reset_request(request):
	if request.method == "POST":
		password_reset_form = PasswordResetForm(request.POST)
		if password_reset_form.is_valid():
			data = password_reset_form.cleaned_data['email']
			associated_users = User.objects.filter(Q(email=data))
			if associated_users.exists():
				for user in associated_users:
					subject = "Password Reset Requested"
					email_template_name = "home/password/password_reset_email.txt"
					c = {
					"email":user.email,
					'domain':'127.0.0.1:8000',
					'site_name': 'Website',
					"uid": urlsafe_base64_encode(force_bytes(user.pk)),
					"user":user,
					'token': default_token_generator.make_token(user),
					'protocol': 'http',
					}
                           
					email = render_to_string(email_template_name, c)
                       
					email = render_to_string(email_template_name, c)
					try:
						send_mail(subject, email, 'dipakgaikwadmg@gmail.com' , [user.email], fail_silently=False)
					except BadHeaderError:
						return HttpResponse('Invalid header found.')
					return redirect ("/password_reset/done/")
	password_reset_form = PasswordResetForm()
	return render(request=request, template_name="home/password/password_reset.html", context={password_reset_form:password_reset_form})


def logout(request):
    auth.logout(request)
    return redirect ('/')



    
    messages.info(request, "You have log in succesfully")
    send_mail(
		'contact form',
		'ti sfd',
		'settings.EMAIL_HOST_USER',s
		['dipakgaikwadmg@gmail.com'],
		
		)
    auth.logout(request)
    return redirect ('/')

def register(request):
	
	if request.method == 'POST':
		first_name = request.POST.get('first_name')
		last_name =request.POST.get('last_name')
		mobile_no =request.POST.get('mobile_no')
		blood_group=request.POST.get('blood_group')
		address =request.POST.get('address')
		gender = request.POST.get('Gender')
		# country = request.POST.get('country')
		state=request.POST.get('state')
		email=request.POST.get('email')
		district=request.POST.get('district')
		dob=request.POST.get('dob')
		adhaar_no=request.POST.get('adhaar_no')
		pincode=request.POST.get('pincode')

		adhar=adhaar_no[-4:]

		username=first_name+last_name+adhar
	

		
		if User.objects.filter(email=email).exists():
			return redirect(reverse('home:register'))
		else:
			user=User.objects.create_user(first_name=first_name,last_name=last_name,email=email,username=username)
			user.save()
			customuser=CustomUser.objects.create(user=user,mobile=mobile_no,blood_grp=blood_group,address=address,gender=gender,state=state,district=district,dob=dob,aadhar=adhaar_no,pincode=pincode)
			customuser.save()
			messages.success(request,"Registration  successful")
			
			print("done")
			
			return redirect(reverse('home:dashboard'))
	else:
			return render(request, 'register.html')
	
def medical(request):
		return render (request,'medical.html')
	


def search(request):
	if request.method == 'POST':
		search = request.POST.get('search')
		user_data=CustomUser.objects.filter(aadhar=search).all()
		return render(request,'search.html',{'user_data':user_data})
	
	
	else:
		return render(request, 'search.html')

def instant_exam(request):
	if request.method == 'POST':
		dob=request.POST.get('dob')
		adhaar_no=request.POST.get('adhaar_no')
		request.session['adhaar_no']=request.POST['adhaar_no']
		if CustomUser.objects.filter(dob=dob,aadhar=adhaar_no).exists():
			obj=CustomUser.objects.filter(dob=dob,aadhar=adhaar_no).all()
			return render(request,'instant_exam/info.html')
		else:
			return redirect(reverse('home:instant_exam'))
	else:
		return render(request,'instant_exam/instant_exam_login.html')

def vitals(request):
	if request.method == 'POST':
		url1=request.session['adhaar_no']
		height=request.POST.get('height')
		weight=request.POST.get('weight')
		bp_sys=request.POST	.get('bp_sys')
		bp_dia=request.POST.get('bp_dia')
		spo2=request.POST.get('spo2')
		pulse=request.POST.get('pulse')
		temp=request.POST.get('temp')
		arm=request.POST.get('arm')

		user_obj=CustomUser.objects.get(aadhar=url1)
		user_obj.height=height
		user_obj.weight=weight
		user_obj.bp_sys=bp_sys
		user_obj.bp_dia=bp_dia
		user_obj.spo2=spo2
		user_obj.pulse=pulse
		user_obj.temp=temp
		user_obj.arm=arm
		user_obj.save()
		print(user_obj)
		return render(request, 'instant_exam/info.html')

	return render(request,'instant_exam/vitals_form.html')

def Issue_No():
    a=random.randint(100000, 1000000)
    today = date.today()
    d = today.strftime("%m%d%Y")
    return str(d)+"-"+str(a)

def Issue_Date():
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    return str(d1)



def symptoms(request):
	if request.method=='POST':
		symptoms_txt = request.POST.get('textarea',None)
		url1=request.session['adhaar_no']

		# Reading the train.csv by removing the
        # last column since it's an empty column
		DATA_PATH = "E:\\Diagnose\\csv\\Training.csv"
		data = pd.read_csv(DATA_PATH).dropna(axis = 1)
        
        # Checking whether the dataset is balanced or not
		disease_counts = data["prognosis"].value_counts()
		temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values
        })

        #Splitting
        # Encoding the target value into numerical
        # value using LabelEncoder
		encoder = LabelEncoder()
		data["prognosis"] = encoder.fit_transform(data["prognosis"])

		X = data.iloc[:,:-1]
		y = data.iloc[:, -1]
		X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size = 0.2, random_state = 24)

        # Defining scoring metric for k-fold cross validation
		def cv_scoring(estimator, X, y):
			return accuracy_score(y, estimator.predict(X))
        
        # Initializing Models
		models = {
            "SVC":SVC(),
            "Gaussian NB":GaussianNB(),
            "Random Forest":RandomForestClassifier(random_state=18)
        }
        
        # Producing cross validation score for the models
		for model_name in models:
			model = models[model_name]
			scores = cross_val_score(model, X, y, cv = 10,
                                    n_jobs = -1,
                                    scoring = cv_scoring)

        # Training and testing SVM Classifier
		svm_model = SVC()
		svm_model.fit(X_train, y_train)
		preds = svm_model.predict(X_test)
		cf_matrix = confusion_matrix(y_test, preds)

        # Training and testing Naive Bayes Classifier
		nb_model = GaussianNB()
		nb_model.fit(X_train, y_train)
		preds = nb_model.predict(X_test)
		cf_matrix = confusion_matrix(y_test, preds)

        # Training and testing Random Forest Classifier
		rf_model = RandomForestClassifier(random_state=18)
		rf_model.fit(X_train, y_train)
		preds = rf_model.predict(X_test)
		cf_matrix = confusion_matrix(y_test, preds)

        # Training the models on whole data
		final_svm_model = SVC()
		final_nb_model = GaussianNB()
		final_rf_model = RandomForestClassifier(random_state=18)
		final_svm_model.fit(X, y)
		final_nb_model.fit(X, y)
		final_rf_model.fit(X, y)
        
        # Reading the test data
		test_data = pd.read_csv("E:\\Diagnose\\csv\\Testing.csv").dropna(axis=1)
        
		test_X = test_data.iloc[:, :-1]
		test_Y = encoder.transform(test_data.iloc[:, -1])
        
        # Making prediction by take mode of predictions
        # made by all the classifiers
		svm_preds = final_svm_model.predict(test_X)
		nb_preds = final_nb_model.predict(test_X)
		rf_preds = final_rf_model.predict(test_X)
        
		final_preds = [mode([i,j,k])[0][0] for i,j,
                    k in zip(svm_preds, nb_preds, rf_preds)]

		symptoms = X.columns.values
    
        # Creating a symptom index dictionary to encode the
        # input symptoms into numerical form
		symptom_index = {}
		for index, value in enumerate(symptoms):
			symptom = " ".join([i.capitalize() for i in value.split("_")])
			symptom_index[symptom] = index
        
		data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
        }
        
        # Defining the Function
        # Input: string containing symptoms separated by commmas
        # Output: Generated predictions by models
		def predictDisease(symptoms):
			symptoms = symptoms.split(",")
            
            # creating input data for the models
			input_data = [0] * len(data_dict["symptom_index"])
			for symptom in symptoms:
				index = data_dict["symptom_index"][symptom]
				input_data[index] = 1
                
            # reshaping the input data and converting it
            # into suitable format for model predictions
			input_data = np.array(input_data).reshape(1,-1)
            
            # generating individual outputs
			rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
			nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
			svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
            
            # making final prediction by taking mode of all predictions
			final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
			predictions = {
                "rf_model_prediction": rf_prediction,
                "naive_bayes_prediction": nb_prediction,
                "svm_model_prediction": nb_prediction,
                "final_prediction":final_prediction
            }
			return predictions
    
        # Testing the function
		prediction = predictDisease(symptoms_txt)

		user_obj=CustomUser.objects.get(aadhar=url1)
		user_obj.symptoms=symptoms_txt
		user_obj.disease=prediction['final_prediction']
		user_obj.save()
		
		print("saved")
		return render(request, 'instant_exam/Report.html', {"result":prediction, "symp":symptoms_txt, "date":d1, "data":user_obj})
	else:
		return render(request, 'instant_exam/symptoms.html')


def directory(request):
	if request.method == 'POST':
		directory = request.POST['directory']
		user_obj=CustomUser.objects.filter(specialization=directory).all()
		return render(request,'display_dir.html',{'user_data':user_obj})
	else:
		return render(request, 'directory.html')

def display(request):
	return render(request, 'instant_exam/display_dir.html')

def survey(request):
	if request.method == 'POST':
		adhar=request.POST['adhar']
		no_of_family_members = request.POST['no_of_family_members']
		name_of_head_of_family = request.POST['name_of_head_of_family']
		age_of_head_of_family = request.POST['age_of_head_of_family']
		no_of_other_adult_males = request.POST['no_of_other_adult_males']
		no_of_other_adult_female = request.POST['no_of_other_adult_female']
		no_of_male_children	= request.POST['no_of_male_children']
		no_of_female_children = request.POST['no_of_female_children']
		water_source = request.POST['water_source']
		water_distance = request.POST['water_distance']
		water_treatment = request.POST['water_treatment']
		non_usage_of_toilets = request.POST['non_usage_of_toilets']
		distance_of_subcentres = request.POST['distance_of_subcentres']
		distance_of_primary_centers = request.POST['distance_of_primary_centers']
		distance_of_community_centers = request.POST['distance_of_community_centers']
		distance_of_district_hospital = request.POST['distance_of_district_hospital']
		distance_of_patahology_lab = request.POST['distance_of_patahology_lab']
		distance_of_medical_store = request.POST['distance_of_medical_store']
		status_of_delivery_of_children = request.POST['status_of_delivery_of_children']
		status_of_vaccination_of_children = request.POST['status_of_vaccination_of_children']
		status_of_female_problems = request.POST['status_of_female_problems']
		centrally_issued_health_insurance = request.POST['centrally_issued_health_insurance']
		state_issued_health_insurance = request.POST['state_issued_health_insurance']
		personal_health_insurance = request.POST['personal_health_insurance']

		if CustomUser.objects.filter(aadhar=adhar).exists():

			survey_data=Survey(adhar=adhar,no_of_family_members=no_of_family_members,name_of_head_of_family=name_of_head_of_family,age_of_head_of_family=age_of_head_of_family,no_of_other_adult_males=no_of_other_adult_males,no_of_other_adult_female=no_of_other_adult_female,no_of_male_children=no_of_male_children,no_of_female_children=no_of_female_children,water_source=water_source,water_distance=water_distance,
			water_treatment = water_treatment,non_usage_of_toilets=non_usage_of_toilets,distance_of_subcentres=distance_of_subcentres,distance_of_primary_centers=distance_of_primary_centers,
			distance_of_community_centers=distance_of_community_centers,distance_of_district_hospital=distance_of_district_hospital,distance_of_patahology_lab=distance_of_patahology_lab,distance_of_medical_store=distance_of_medical_store,status_of_delivery_of_children=status_of_delivery_of_children,status_of_vaccination_of_children=status_of_vaccination_of_children,status_of_female_problems=status_of_female_problems,centrally_issued_health_insurance=centrally_issued_health_insurance,state_issued_health_insurance=state_issued_health_insurance,personal_health_insurance=personal_health_insurance)

			survey_data.save()
			messages.success(request,"Survey Completted successfully")
			return redirect(reverse('home:dashboard')) 
		else:
			return render(request, 'survey/survey.html')		

	else:
		return render(request, 'survey/survey.html')	
def raise_query(request):
	if request.method == 'POST':
		Issue_no=Issue_No()
		Issue_date=Issue_Date()
		subject = request.POST['subject']
		issue=request.POST['issue']
		raise_data=CustomUser.objects.get(user_id=request.user.id)
		raise_data.Issue_no=Issue_no
		raise_data.Issue_date=Issue_date
		raise_data.subject =subject
		raise_data.issue=issue
		raise_data.save()
		messages.success(request,"Issues Submitted successfully")

		return redirect(reverse('home:dashboard'))
	else:
		return render(request, 'raise_query/raise_issue.html')
	



	
def profile(request):
	
	data=CustomUser.objects.get(user_id=request.user.id)

	return render(request, 'profile.html', {'user_data':data})

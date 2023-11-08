import requests


url = 'http://localhost:9696/predict'

#patient_id = 'xyz-123'
unhealthy_patient = {
	"age": 56,
#	"sex": 1,
	"cholesterol": 500,
	"heart_rate": 90,
#	"diabetes": 1,
	"family_history": 1,
	"smoking": 1,
#	"alcohol_consumption": 1,
#	"exercise_hours_per_week": 1,
#	"previous_heart_problems": 1,
#	"medication_use": 1,
#	"stress_level": 6,
#	"sedentary_hours_per_day": 14,
#	"bmi": 30.4,
	"triglycerides": 220,
#	"sleep_hours_per_day": 6,
 	"blood_pressure_sistolic":  200,
#	"blood_pressure_diastolic": 90,
#	"diet": "unhealthy"
}

#patient_id = 'xyz-200'
healthy_patient = {
	"age": 29,
#	"sex": 0,
	"cholesterol": 120,
	"heart_rate": 60,
#	"diabetes": 0,
	"family_history": 0,
	"smoking": 0,
#	"alcohol_consumption": 1,
#	"exercise_hours_per_week": 10,
#	"previous_heart_problems": 0,
#	"medication_use": 0,
#	"stress_level": 6,
#	"sedentary_hours_per_day": 5,
#	"bmi": 24.0,
	"triglycerides": 140,
# 	"sleep_hours_per_day": 8,
 	"blood_pressure_sistolic":  110,
#	"blood_pressure_diastolic": 60,
#	"diet": "healthy"
}


print("Test Patient Data: ", unhealthy_patient)
response = requests.post(url, json=unhealthy_patient).json()
print(response)


print("Test Patient Data: ", healthy_patient)
response = requests.post(url, json=healthy_patient).json()
print(response)


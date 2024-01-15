#Advait Kedar Joshi (advaitkjoshi@gmail.com, +91 7249252910)
#Problem Statement: Marathi Text Classification
#To fine tune the named entity recognition in mahaNLP to detect Date, Month, Year and Time
#Future scope: Use Google Calendar API to automatically create event, extracting date and time from Marathi sentence

from mahaNLP.tagger import EntityRecognizer
from mahaNLP.tokenizer import Tokenize
import pandas as pd
import csv

#determine whether the token is integer or not
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#determine whether the token in a month or not
def is_month(word):
    month_list = ['जानेवारी','फेब्रुवारी','मार्च','एप्रिल','मे','जून','जुलै','ऑगस्ट','सप्टेंबर','ऑक्टोबर','नोव्हेंबर','डिसेंबर']
    if word in month_list:
        return True
    return False

#determine whether the token is a day or not
def is_day(word):
    day_list = ["रविवार","रविवारी", "सोमवार","सोमवारी", "मंगळवार","मंगळवारी", "बुधवार","बुधवारी", "गुरुवार","गुरुवारी", "शुक्रवार","शुक्रवारी", "शनिवार","शनिवारी"]
    if word in day_list:
        return True
    return False

#if any of the three words namely: 'रात्री','संध्याकाळी','सायंकाळी' is detected, time is considered as PM, otherwise AM
def is_pm(word):
    pm_list = ['रात्री','संध्याकाळी','सायंकाळी']
    if word in pm_list:
        return True
    return False

#creating entity recongition model
model = EntityRecognizer()
token_model = Tokenize()

#Prototype phase testing on few sentences
#Few sample sentences with date and time details
text1 = 'मला आदित्यला शुक्रवार,दिनांक 20 ऑक्टोबर 2023 ला रात्री 10 वाजता भेटायचे आहे.मंगळवार दिनांक 24 मार्च रोजी परीक्षा सुरू होईल.'
text2 = 'मंगळवार दिनांक 24 मार्च रोजी परीक्षा सुरू होईल.'
text3 = 'परीक्षेची वेळ सकाळी 10 ते सायंकाळी 7 अशी आहे.'
text4 = 'मी 19 नोव्हेंबर ते 2 डिसेंबर पर्यंत सुट्टीवर असेन.'
text5 = 'आज बुधवार आहे, दिनांक 16 नोव्हेंबर 2023. वेळ 2.'
text6 = ' नव्या सप्ताहाची सुरूवात झाली, आज रविवार आहे, तारीख 26 जून 2023.'

dataset = pd.read_csv("marathidataset copy.csv")

for index, row in dataset.iterrows():
#tokenizing the sentence
    token_list = model.get_token_labels(row['sentence'], as_dict=True)

    timing = [item['word'] for item in token_list if item['entity_group'] == 'Time']    #grouped under 'Time'
    dates = [item['word'] for item in token_list if item['entity_group'] == 'Date']     #grouped under 'Date'
    persons = [item['word'] for item in token_list if item['entity_group'] == 'Person'] #grouped under 'Person'

    result = []

    for item in dates:
        if(is_day(item)):
            result.append("Day: "+item)

    for item in dates:
        if(is_integer(item)):
            if(0<int(item)<=31):
                result.append("Date: "+item)
            elif(1000<=int(item)<=3000):    #targetting calendar events in this year(2023) and next year(2024)
                result.append("Year: "+item)
        elif(is_month(item)):
            result.append("Month: "+item)

    for i in range(len(timing)):
        item = timing[i]
        if(is_integer(item)):
            if(1<=int(item)<=12):             #checking if value lies in range of 1 to 12
                if(is_pm(timing[i-1])):         #checking if it is preceeded by a word which implies PM
                    result.append("Time: "+item+"PM")
                else:
                    result.append("Time: "+item+"AM")

    with open("new_resultfile.csv", 'a', newline='') as file:

        csv_writer = csv.writer(file)
        csv_writer.writerow(result)
    print(result)

#Successful classification of text under:
#1. Day
#2. Date
#3. Month
#4. Year
#5. Time (with AM and PM)

#Future prospects: To apply lemmatization for processing complex Marathi sentences. Incorporate Google Calendar API.
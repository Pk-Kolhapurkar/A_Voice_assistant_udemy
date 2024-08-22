import datetime
import pandas as pd

current_time = datetime.datetime.now()
#print(current_time)
current_time, current_minute = datetime.datetime.time(current_time).hour, datetime.datetime.time(current_time).minute
#print('Current time:', current_time)
#print('Current minute: ', current_minute)
current_date = datetime.datetime.date(datetime.datetime.today())
#print('Current date:', current_date)

agenda_worksheet = '/Users/jonesgranatyr/Documents/Ensino/IA Expert/Cursos/Virtual assistent/virtual_assistant/agenda.xlsx'
agenda = pd.read_excel(agenda_worksheet)
#print(agenda)

description, responsible, hour_agenda = [], [], []
for index, row in agenda.iterrows():
    #print(index)
    #print(row)
    date = datetime.datetime.date(row['date'])
    #print(date)
    complete_hour = datetime.datetime.strptime(str(row['hour']), '%H:%M:%S')
    #print(complete_hour)
    hour = datetime.datetime.time(complete_hour).hour
    #print(hour)

    if current_date == date:
        if hour >= current_time:
            description.append(row['description'])
            responsible.append(row['responsible'])
            hour_agenda.append(row['hour'])

#print(description)
#print(responsible)
#print(hour_agenda)

def load_agenda():
    if description:
        return description, responsible, hour_agenda
    else:
        return False

#print(load_agenda())
















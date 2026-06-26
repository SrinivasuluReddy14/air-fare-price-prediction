from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import io
import base64

from .models import Signup as SignupModel

global uname, le1, le2, le3, scaler, rf, dataset, X, Y

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def SignupPage(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def SignupAction(request):
    if request.method == 'POST':

        username = request.POST.get('t1')
        password = request.POST.get('t2')
        contact = request.POST.get('t3')
        email = request.POST.get('t4')
        address = request.POST.get('t5')

        if SignupModel.objects.filter(username=username).exists():
            return render(
                request,
                'Signup.html',
                {'data': 'Username already exists'}
            )

        SignupModel.objects.create(
            username=username,
            password=password,
            contact_no=contact,
            email_id=email,
            address=address
        )

        return render(
            request,
            'Signup.html',
            {'data': 'Signup Successful'}
        )

def UserLoginAction(request):

    global uname

    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        user = SignupModel.objects.filter(
            username=username,
            password=password
        ).first()

        if user:
            uname = username
            request.session['username'] = username

            return render(
                request,
                'UserScreen.html',
                {
                    'data': f'Welcome {username}'
                }
            )

        return render(
            request,
            'UserLogin.html',
            {
                'data': 'Invalid Login Details'
            }
        )

    return render(request, 'UserLogin.html')

def DatasetCollection(request):
    if request.method == 'GET':
        global dataset
        dataset = pd.read_csv("Dataset/FlightPrice.csv", usecols=['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time', 'Duration', 'Price'])
        columns = dataset.columns
        data = dataset.values
        output = '<table border=1><tr>'
        for i in range(len(columns)):
            output+='<th><font size="" color="black">'+columns[i]+'</th>'
        output += "</tr>"    
        for i in range(len(data)):
            output+='<tr><td><font size="" color="black">'+str(data[i,0])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,1])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,2])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,3])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,4])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,5])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,6])+'</td>'
            output+='<td><font size="" color="black">'+str(data[i,7])+'</td></tr>'
        output += "</table><br/>"
        airlines = np.unique(dataset['Airline'])
        temp = dataset[['Airline', 'Price', 'Source', 'Destination']]
        plt.figure(figsize=(12,6))

        sns.pointplot(
            x="Airline",
            y="Price",
            hue="Source",
            data=temp
        )

        plt.title("Price Comparison between Different Airlines")
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        output += "</table><br/>"
        context= {'data':output, 'graph1b64': img_b64}
        return render(request, 'UserScreen.html', context)        

def DatasetCleaning(request):

    global dataset, le1, le2, le3, scaler, X, Y

    if 'dataset' not in globals() or dataset is None:

        return render(
            request,
            'UserScreen.html',
            {'data':'Please click Dataset Collection first'}
        )

    if 'Date_of_Journey' not in dataset.columns:

        return render(
            request,
            'UserScreen.html',
            {
                'data':
                'Dataset already cleaned. Click Dataset Collection again before cleaning.'
            }
        )

    if request.method == 'GET':

        dataset['Date_of_Journey'] = pd.to_datetime(
            dataset['Date_of_Journey'],
            dayfirst=True
        )

        dataset['year'] = dataset['Date_of_Journey'].dt.year
        dataset['month'] = dataset['Date_of_Journey'].dt.month
        dataset['day'] = dataset['Date_of_Journey'].dt.day

        Y = dataset['Price'].to_numpy()

        dataset.drop(
            [
                'Date_of_Journey',
                'Dep_Time',
                'Arrival_Time',
                'Duration',
                'Price'
            ],
            axis=1,
            inplace=True,
            errors='ignore'
        )

        le1 = LabelEncoder()
        le2 = LabelEncoder()
        le3 = LabelEncoder()

        dataset['Airline'] = le1.fit_transform(
            dataset['Airline'].astype(str)
        )

        dataset['Source'] = le2.fit_transform(
            dataset['Source'].astype(str)
        )

        dataset['Destination'] = le3.fit_transform(
            dataset['Destination'].astype(str)
        )

        print("Airlines:")
        print(le1.classes_)

        print("Source Cities:")
        print(le2.classes_)

        print("Destination Cities:")
        print(le3.classes_)

        scaler = MinMaxScaler()

        X = scaler.fit_transform(dataset.values)

        columns = dataset.columns

        output = '<table border=1><tr>'

        for col in columns:
            output += f'<th>{col}</th>'

        output += '</tr>'

        for row in X:

            output += '<tr>'

            for value in row:
                output += f'<td>{value}</td>'

            output += '</tr>'

        output += '</table>'

        return render(
            request,
            'UserScreen.html',
            {'data': output}
        )
    

def TrainRF(request):

    if request.method == 'GET':

        global X, Y, rf

        if 'X' not in globals():

            return render(
                request,
                'UserScreen.html',
                {'data':'Please run Dataset Cleaning first'}
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=0
        )

        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=0
        )

        rf.fit(X_train, y_train)
        predict = rf.predict(X_test)
        output = '<table border=1><tr>'
        output+='<tr><th><font size="" color="black">True Test Price</th><th><font size="" color="black">Random Forest Predicted Price</th></tr>'

        labels = y_test[0:100]
        predict = predict[0:100]
        for i in range(len(labels)):
            output+='<tr><td><font size="" color="black">'+str(labels[i])+'</td>'
            output+='<td><font size="" color="black">'+str(int(predict[i]))+'</td></tr>'
        output += "</table><br/><br/><br/>"
        plt.plot(labels, color = 'red', label = 'True Test Price')
        plt.plot(predict, color = 'green', label = 'Random Forest Predicted Price')
        plt.title("Random Forest Flight Price Prediction Graph")
        plt.xlabel('Number of Days')
        plt.ylabel('Predicted Prices')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        output += "</table><br/>"
        context= {'data':output, 'graph1b64': img_b64}
        return render(request, 'UserScreen.html', context)   

def PredictPrices(request):
    if request.method == 'GET':
        output = '''
        <tr>
        <td>Travel Date</td>
        <td>
        <select name="t1" required>
        
        <option value="">
        Select Day
        </option>
        '''
        for i in range(1, 32):
            if i < 10:
                output += '<option value="0'+str(i)+'">0'+str(i)+"</option>"
            else:
                output += '<option value="'+str(i)+'">'+str(i)+"</option>"
        output += '''
        </select>
        
        &nbsp;
        
        <select name="t2" required>
        
        <option value="">
        Select Month
        </option>
        '''
        for i in range(1, 13):
            if i < 10:
                output += '<option value="0'+str(i)+'">0'+str(i)+"</option>"
            else:
                output += '<option value="'+str(i)+'">'+str(i)+"</option>"
        output += '''
        </select>
        
        &nbsp;
        
        <select name="t3" required>
        <option value="">
        Select Year
        </option>
        '''
        for i in range(2023, 2050):
            output += '<option value="'+str(i)+'">'+str(i)+"</option>"
        output += '</select></td></tr>'
        context= {'data1': output}
        return render(request, 'PredictPrices.html', context)

def PredictPricesAction(request):
    if 'rf' not in globals():
        return render(
            request,
            'UserScreen.html',
            {'data':'Please run Dataset Collection, Dataset Cleaning and Train Model first'}
            )
    if request.method == 'POST':
        global le1, le2, le3, scaler, rf
        date = request.POST.get('journey_date')
        airline = request.POST.get('t4', False)
        source = request.POST.get('t5', False)
        dest = request.POST.get('t6', False)
        data = []
        data.append([airline, date, source, dest])
        data = pd.DataFrame(data, columns=['Airline', 'Date_of_Journey', 'Source', 'Destination'])
        data['Date_of_Journey'] = pd.to_datetime(
            data['Date_of_Journey']
        )
        data['year'] = data['Date_of_Journey'].dt.year
        data['month'] = data['Date_of_Journey'].dt.month
        data['day'] = data['Date_of_Journey'].dt.day

        if source not in le2.classes_:
            return render(
                request,
                'UserScreen.html',
                {'data':'Invalid Source City'}
                )
            
            if dest not in le3.classes_:
                return render(
                    request,
                    'UserScreen.html',
        {'data':'Invalid Destination City'}
    )

        data['Airline'] = pd.Series(le1.transform(data['Airline'].astype(str)))#encode all str columns to numeric
        data['Source'] = pd.Series(le2.transform(data['Source'].astype(str)))#encode all str columns to numeric
        data['Destination'] = pd.Series(le3.transform(data['Destination'].astype(str)))#encode all str columns to numeric
        data.drop(['Date_of_Journey'], axis = 1,inplace=True)
        X1 = data.values
        X1 = scaler.transform(X1)
        predict = rf.predict(X1)[0]
        output = "Predicted Flight Price = "+str(predict)
        context= {'data': output}
        return render(request, 'UserScreen.html', context)
        









        

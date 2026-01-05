from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import io
import base64

global uname, le1, le2, le3, scaler, rf, dataset, X, Y

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'FlightPrice',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from signup where username = '"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == email:
                    status = 'Given Username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'FlightPrice',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email_id,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data':status}
        return render(request, 'Signup.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        option = 0
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'FlightPrice',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    uname = username
                    option = 1
                    break
        if option == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)

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
        sns.catplot(x="Airline", y="Price", hue='Source', data=temp, kind='point')
        plt.title("Price Comparison between Different Airlines")
        plt.xticks(rotation=90)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        output += "</table><br/>"
        context= {'data':output, 'graph1b64': img_b64}
        return render(request, 'UserScreen.html', context)        

def DatasetCleaning(request):
    if request.method == 'GET':
        global dataset, le1, le2, le3, scaler, X, Y
        dataset['Date_of_Journey'] = pd.to_datetime(dataset['Date_of_Journey'])
        dataset['year'] = dataset['Date_of_Journey'].dt.year
        dataset['month'] = dataset['Date_of_Journey'].dt.month
        dataset['day'] = dataset['Date_of_Journey'].dt.day
        Y = dataset['Price'].ravel()
        dataset.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Price'], axis = 1,inplace=True)
        le1 = LabelEncoder()
        le2 = LabelEncoder()
        le3 = LabelEncoder()
        dataset['Airline'] = pd.Series(le1.fit_transform(dataset['Airline'].astype(str)))#encode all str columns to numeric
        dataset['Source'] = pd.Series(le2.fit_transform(dataset['Source'].astype(str)))#encode all str columns to numeric
        dataset['Destination'] = pd.Series(le3.fit_transform(dataset['Destination'].astype(str)))#encode all str columns to numeric
        scaler = MinMaxScaler()
        columns = dataset.columns
        X = dataset.values
        X = scaler.fit_transform(X)
        output = '<table border=1><tr>'
        for i in range(len(columns)):
            output+='<th><font size="" color="black">'+columns[i]+'</th>'
        output += "</tr>"    
        for i in range(len(X)):
            output+='<tr><td><font size="" color="black">'+str(X[i,0])+'</td>'
            output+='<td><font size="" color="black">'+str(X[i,1])+'</td>'
            output+='<td><font size="" color="black">'+str(X[i,2])+'</td>'
            output+='<td><font size="" color="black">'+str(X[i,3])+'</td>'
            output+='<td><font size="" color="black">'+str(X[i,4])+'</td>'
            output+='<td><font size="" color="black">'+str(X[i,5])+'</td></tr>'
        output += "</table><br/><br/><br/>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def TrainRF(request):
    if request.method == 'GET':
        global dataset, le1, le2, le3, scaler, X, Y, rf
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        rf = RandomForestRegressor()
        rf.fit(X, Y)
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
        output = '<tr><td><font size="" color="black">Travel&nbsp;Date</b></td><td><select name="t1">'
        for i in range(1, 32):
            if i < 10:
                output += '<option value="0'+str(i)+'">0'+str(i)+"</option>"
            else:
                output += '<option value="'+str(i)+'">'+str(i)+"</option>"
        output += '</select>&nbsp;<select name="t2">'
        for i in range(1, 13):
            if i < 10:
                output += '<option value="0'+str(i)+'">0'+str(i)+"</option>"
            else:
                output += '<option value="'+str(i)+'">'+str(i)+"</option>"
        output += '</select>&nbsp;<select name="t3">'
        for i in range(2023, 2050):
            output += '<option value="'+str(i)+'">'+str(i)+"</option>"
        output += '</select></td></tr>'
        context= {'data1': output}
        return render(request, 'PredictPrices.html', context)

def PredictPricesAction(request):
    if request.method == 'POST':
        global le1, le2, le3, scaler, rf
        dd = request.POST.get('t1', False)
        mm = request.POST.get('t2', False)
        yy = request.POST.get('t3', False)
        date = dd+"/"+mm+"/"+yy
        airline = request.POST.get('t4', False)
        source = request.POST.get('t5', False)
        dest = request.POST.get('t6', False)
        data = []
        data.append([airline, date, source, dest])
        data = pd.DataFrame(data, columns=['Airline', 'Date_of_Journey', 'Source', 'Destination'])
        data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])
        data['year'] = data['Date_of_Journey'].dt.year
        data['month'] = data['Date_of_Journey'].dt.month
        data['day'] = data['Date_of_Journey'].dt.day
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
        









        

from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Signup', views.Signup, name="Signup"),
	       path('SignupAction', views.SignupAction, name="SignupAction"),
	       path('DatasetCollection', views.DatasetCollection, name="DatasetCollection"),
	       path('DatasetCleaning', views.DatasetCleaning, name="DatasetCleaning"),	      
	       path('TrainRF', views.TrainRF, name="TrainRF"),
	       path('PredictPrices', views.PredictPrices, name="PredictPrices"),
	       path('PredictPricesAction', views.PredictPricesAction, name="PredictPricesAction"),
]

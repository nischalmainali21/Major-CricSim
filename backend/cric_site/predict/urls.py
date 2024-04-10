from django.urls import path
from .views import PredictionAPIView, PlotDataAPIView
from .views2 import MatchSimulationAPIView
urlpatterns = [
    path('full_data/', PredictionAPIView.as_view(), name='full_data'),
    path('plot_data/', PlotDataAPIView.as_view(), name='plot_data'),
    path('team-api/',MatchSimulationAPIView.as_view(), name = 'team-api/')
]

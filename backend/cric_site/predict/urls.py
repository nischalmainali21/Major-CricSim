from django.urls import path
from .views import PredictionAPIView, PlotDataAPIView

urlpatterns = [
    path('full_data/', PredictionAPIView.as_view(), name='full_data'),
    path('plot_data/', PlotDataAPIView.as_view(), name='plot_data'),
]

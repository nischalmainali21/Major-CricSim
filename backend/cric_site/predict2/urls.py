from django.urls import path
from .views import ListBatsmen, ListBowlers, ListPlayers, ListVenues, CreateMatchAPIView

urlpatterns = [
    path('player_data/', ListPlayers.as_view(), name='player_data'),
    path('venue_data/', ListVenues.as_view(), name='venue_data'),
    path('batsman_data/', ListBatsmen.as_view(), name='batsman_data'),
    path('bowler_data/', ListBowlers.as_view(), name='bowler_data'),
    path('create_match/', CreateMatchAPIView.as_view(), name='create_match')
    
]

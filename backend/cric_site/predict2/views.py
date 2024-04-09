from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Player, Venue, Batsman, Bowler
from .serializers import PlayerSerializer, VenueSerializer, BatsmanSerializer, BowlerSerializer, MatchSerializer

class ListPlayers(APIView):
    """
    API endpoint to list all players.
    """
    def get(self, request, format=None):
        players = Player.objects.all()
        serializer = PlayerSerializer(players, many=True)
        return Response(serializer.data)


class ListVenues(APIView):
    """
    API endpoint to list all venues.
    """
    def get(self, request, format=None):
        venues = Venue.objects.all()
        serializer = VenueSerializer(venues, many=True)
        return Response(serializer.data)
    
class ListBatsmen(APIView):
    """
    API endpoint to list all venues.
    """
    def get(self, request, format=None):
        venues = Batsman.objects.all()
        serializer = BatsmanSerializer(venues, many=True)
        return Response(serializer.data)
    
class ListBowlers(APIView):
    """
    API endpoint to list all venues.
    """
    def get(self, request, format=None):
        venues = Bowler.objects.all()
        serializer = BowlerSerializer(venues, many=True)
        return Response(serializer.data)
    
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class CreateMatchAPIView(APIView):
    def post(self, request, format=None):
        serializer = MatchSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    


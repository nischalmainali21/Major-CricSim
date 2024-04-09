from rest_framework import serializers
from .models import Player, Venue, Batsman, Bowler, Match

class PlayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Player
        fields = '__all__'

class VenueSerializer(serializers.ModelSerializer):
    class Meta:
        model = Venue
        fields = '__all__'

class BatsmanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Batsman
        fields = '__all__'

class BowlerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bowler
        fields = '__all__'

from rest_framework import serializers

class MatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = Match
        fields = '__all__'

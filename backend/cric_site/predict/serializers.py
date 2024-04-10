from rest_framework import serializers

class TeamSerializer(serializers.Serializer):
    team1 = serializers.ListField(child=serializers.CharField())
    team2 = serializers.ListField(child=serializers.CharField())
    venue_name = serializers.CharField()

    def validate(self, data):
        if 'team1' not in data or 'team2' not in data or 'venue_name' not in data:
            raise serializers.ValidationError("Both teams as well as venue must be provided.")
        if len(data['team1']) != 11 or len(data['team2']) != 11:
            raise serializers.ValidationError("Each team must have exactly 11 players.")
        return data
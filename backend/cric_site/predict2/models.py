from django.db import models

class Player(models.Model):
    name = models.CharField(max_length=100)
    player_type = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name

class Venue(models.Model):
    venue_name = models.CharField(max_length=100)
    matches = models.IntegerField()
    total_run = models.IntegerField()
    dot_balls = models.IntegerField()
    is_wicket_delivery = models.IntegerField()
    delivery = models.IntegerField()
    fours = models.IntegerField()
    sixes = models.IntegerField()
    average_score = models.FloatField()
    dot_ball_percentage = models.FloatField()
    average_wickets_fallen = models.FloatField()
    boundary_frequency = models.FloatField()
    
    def __str__(self) -> str:
        return self.venue_name

class Batsman(Player):
    team = models.CharField(max_length=100)
    active_ratio_death = models.FloatField()
    striking_ratio_death = models.FloatField()
    true_sr_death = models.FloatField()
    true_avg_death = models.FloatField()
    active_ratio_middle = models.FloatField()
    striking_ratio_middle = models.FloatField()
    true_sr_middle = models.FloatField()
    true_avg_middle = models.FloatField()
    active_ratio_powerplay = models.FloatField()
    striking_ratio_powerplay = models.FloatField()
    true_sr_powerplay = models.FloatField()
    true_avg_powerplay = models.FloatField()
    
    def __str__(self):
        return self.name

class Bowler(Player):
    team = models.CharField(max_length=100)
    true_economy_death = models.FloatField()
    true_sr_death = models.FloatField()
    dot_percentage_death = models.FloatField()
    containment_death = models.FloatField()
    true_economy_middle = models.FloatField()
    true_sr_middle = models.FloatField()
    dot_percentage_middle = models.FloatField()
    containment_middle = models.FloatField()
    true_economy_powerplay = models.FloatField()
    true_sr_powerplay = models.FloatField()
    dot_percentage_powerplay = models.FloatField()
    containment_powerplay = models.FloatField()
    
    def __str__(self):
        return self.name
    
class Match(models.Model):
    venue = models.ForeignKey(Venue, on_delete=models.CASCADE)
    team1_batters = models.ManyToManyField(Batsman, related_name='team1_matches')
    team2_batters = models.ManyToManyField(Batsman, related_name='team2_matches')
    team1_bowlers = models.ManyToManyField(Bowler, related_name='team1_matches')
    team2_bowlers = models.ManyToManyField(Bowler, related_name='team2_matches')



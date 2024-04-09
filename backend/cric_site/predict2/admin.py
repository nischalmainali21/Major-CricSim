from django.contrib import admin
from .models import Player, Batsman, Bowler, Venue

# Register your models here.
admin.site.register([Player, Batsman, Bowler, Venue])
# import_venues.py

import csv
from django.core.management.base import BaseCommand
from predict2.models import Venue  # Import your Venue model here

class Command(BaseCommand):
    help = 'Import venues from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', help='/Users/prakash/Desktop/all_data/venue.csv')

    def handle(self, *args, **options):
        csv_file = options['csv_file']

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                Venue.objects.create(
                    venue_name=row['venue_name'],
                    matches=row['Matches'],
                    total_run=row['total_run'],
                    dot_balls=row['Dot_Balls'],
                    is_wicket_delivery=row['isWicketDelivery'],
                    delivery=row['delivery'],
                    fours=row['Fours'],
                    sixes=row['Sixes'],
                    average_score=row['Average Score'],
                    dot_ball_percentage=row['Dot Ball Percentage'],
                    average_wickets_fallen=row['Average Wickets Fallen'],
                    boundary_frequency=row['Boundary Frequency']
                )

        self.stdout.write(self.style.SUCCESS('Venues imported successfully!'))

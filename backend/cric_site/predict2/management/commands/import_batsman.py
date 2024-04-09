# import_batters.py

import csv
from django.core.management.base import BaseCommand
from predict2.models import Batsman  # Import your Batsman model here

class Command(BaseCommand):
    help = 'Import batters from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', help='/Users/prakash/Desktop/all_data/all_bat.csv')

    def handle(self, *args, **options):
        csv_file = options['csv_file']

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                Batsman.objects.create(
                    name=row['batter'],
                    # team=row['team'],
                    active_ratio_death=row['Active Ratio_death'],
                    striking_ratio_death=row['Striking Ratio_death'],
                    true_sr_death=row['True SR_death'],
                    true_avg_death=row['True Avg_death'],
                    active_ratio_middle=row['Active Ratio_middle'],
                    striking_ratio_middle=row['Striking Ratio_middle'],
                    true_sr_middle=row['True SR_middle'],
                    true_avg_middle=row['True Avg_middle'],
                    active_ratio_powerplay=row['Active Ratio_powerplay'],
                    striking_ratio_powerplay=row['Striking Ratio_powerplay'],
                    true_sr_powerplay=row['True SR_powerplay'],
                    true_avg_powerplay=row['True Avg_powerplay']
                )

        self.stdout.write(self.style.SUCCESS('Batters imported successfully!'))

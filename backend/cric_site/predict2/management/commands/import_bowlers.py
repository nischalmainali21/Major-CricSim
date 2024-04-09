# import_bowlers.py

import csv
from django.core.management.base import BaseCommand
from predict2.models import Bowler  # Import your Bowler model here

class Command(BaseCommand):
    help = 'Import bowlers from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', help='/Users/prakash/Desktop/all_data/all_bowl.csv')

    def handle(self, *args, **options):
        csv_file = options['csv_file']

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                Bowler.objects.create(
                    name=row['bowler'],
                    # team=row['team'],
                    true_economy_death=row['True Economy_death'],
                    true_sr_death=row['True SR_death'],
                    dot_percentage_death=row['dotpercentage_death'],
                    containment_death=row['Containment_death'],
                    true_economy_middle=row['True Economy_middle'],
                    true_sr_middle=row['True SR_middle'],
                    dot_percentage_middle=row['dotpercentage_middle'],
                    containment_middle=row['Containment_middle'],
                    true_economy_powerplay=row['True Economy_powerplay'],
                    true_sr_powerplay=row['True SR_powerplay'],
                    dot_percentage_powerplay=row['dotpercentage_powerplay'],
                    containment_powerplay=row['Containment_powerplay']
                )

        self.stdout.write(self.style.SUCCESS('Bowlers imported successfully!'))

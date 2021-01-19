"""
Module with general helper functions for the DLMP analysis
"""

import random
import datetime
import numpy as np
import pandas as pd

import fledge.data_interface

class Random(object):

    @staticmethod
    def sample_and_remove(
            population_list: list,
            num_of_samples: int = 1
    ) -> [list, list]:
        samples = random.sample(population_list, k=num_of_samples)
        new_population_list = population_list.copy().remove(samples)
        return [samples, new_population_list]

    @staticmethod
    def sample(
            population_list: list,
            num_of_samples: int = 1
    ) -> [list, list]:
        return random.sample(population_list, k=num_of_samples)


class Calendar(object):
    seasons_monday_start: dict = {}
    leap_year = 2000  # Dummy leap_year to include 29 of Feb
    seasons: list

    def __init__(self):

        # According to German seasons
        season_start_dates = {
            'spring': datetime.date(self.leap_year, 3, 1),
            'summer': datetime.date(self.leap_year, 6, 1),
            'fall': datetime.date(self.leap_year, 9, 1),
            'winter_1': datetime.date(self.leap_year, 12, 1),
            'winter_2': datetime.date(self.leap_year, 1, 1)
        }
        season_end_dates = {
            'spring': datetime.datetime(self.leap_year, 5, 31, 23, 59, 59),
            'summer': datetime.datetime(self.leap_year, 8, 31, 23, 59, 59),
            'fall': datetime.datetime(self.leap_year, 11, 30, 23, 59, 59),
            'winter_1': datetime.datetime(self.leap_year, 12, 31, 23, 59, 59),
            'winter_2': datetime.datetime(self.leap_year, 2, 29, 23, 59, 59)
        }

        self.seasons = [('winter', (season_start_dates['winter_2'], season_end_dates['winter_2'].date())),
                        ('spring', (season_start_dates['spring'], season_end_dates['spring'].date())),
                        ('summer', (season_start_dates['summer'], season_end_dates['summer'].date())),
                        ('fall', (season_start_dates['fall'], season_end_dates['fall'].date())),
                        ('winter', (season_start_dates['winter_1'], season_end_dates['winter_1'].date()))]

        # Adjust all dates so that every season starts on a Monday (for compatibility with fledge)
        for date in season_start_dates:
            if 'winter_2' not in date:
                season_start_dates[date] = season_start_dates[date] + datetime.timedelta(
                    days=-season_start_dates[date].weekday())
            else:
                # Here we have to cut the first days of the year before the first Monday
                season_start_dates[date] = season_start_dates[date] + datetime.timedelta(
                    days=-season_start_dates[date].weekday(), weeks=1)

        for date in season_end_dates:
            if season_end_dates[date].weekday() != 6:
                season_end_dates[date] = season_end_dates[date] + datetime.timedelta(
                    days=-season_end_dates[date].weekday() - 1)

        for season in season_start_dates:
            if 'winter' not in season:
                self.seasons_monday_start[season] = pd.date_range(start=season_start_dates[season], end=season_end_dates[season],
                                                freq='T'),  # 'T' for minutely intervals
            else:
                self.seasons_monday_start['winter'] = pd.date_range(start=season_start_dates['winter_2'],
                                                  end=season_end_dates['winter_2'],
                                                  freq='T').append(
                    pd.date_range(start=season_start_dates['winter_1'], end=season_end_dates['winter_1'], freq='T'))

    def get_season(
            self,
            date_time
    ):
        date_time = date_time.date()
        # we don't really care about the actual year so replace it with our dummy leap_year
        date_time = date_time.replace(year=self.leap_year)
        # return season our date falls in.
        return next(season for season, (start, end) in self.seasons if start <= date_time <= end)

    def get_season_of_scenario_data(
            self,
            scenario_data: fledge.data_interface.ScenarioData
    ) -> [str, list]:
        # The main difference between this method and get_season is that it returns it will change all season starts to
        # a Monday so scenario data is not split between seasons
        # TODO: there is probably a more performant solution than this (compare to get_season())

        for season in self.seasons.keys():
            dates_in_season = np.array(self.seasons[season])
            if scenario_data.timesteps[0] in dates_in_season:
                break

        if season is None:
            season = 'winter'

        time_format = '%W'
        weeknums = self.seasons[season].strftime(time_format).unique().to_list()

        return [season, weeknums]
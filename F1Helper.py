import fastf1 as ff1
import requests
import json
import pandas as pd
import numpy as np
import math
import os
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageColor

from matplotlib.pyplot import imshow
from fastf1.core import Laps, Lap
from typing import Union
from datetime import timedelta


class F1Helper:

    def __init__(self, season: int = None, grandprix: str = None):
        self.season = 0
        self.event = None
        self.events = pd.DataFrame()
        self.drivers = pd.DataFrame()
        self.sessions = {}
        if not os.path.exists('cache'):
            os.makedirs('cache')
        ff1.Cache.enable_cache('cache')
        if season:
            self.set_season(season)
        if grandprix:
            self.set_default_event(grandprix)

    def set_season(self, year: int) -> None:
        """
        Sets the sesason. All references to a grandprix
        will be made in this season. If the season is not within the
        range 1952 to 2022, the season is set to 2021.
        Args:
            year: int, specifying the year

        Returns:
            None
        """
        if isinstance(year, int) and 1950 < year < 2022:
            new_season = year
        else:
            new_season = 2021
        if new_season != self.season:
            self.season = new_season
            self.events = pd.DataFrame()
            self.drivers = pd.DataFrame()
            self.sessions = {}

    def get_season(self) -> int:
        """
        Returns the set season
        Returns:
            Season set (int)
        """
        return self.season

    def get_events(self) -> pd.DataFrame:
        """
        Return a dataframe with a list of all events in the selected season

        Columns: ['season', 'round', 'raceName', 'date', 'time', 'circuitId', 'circuitName',
                  'lat', 'long', 'locality', 'country']

        Returns:
            Dataframe containing all events of the season
        """
        if not self.events.empty:
            return self.events
        if self.season < 1950 or self.season > 2030:
            return None
        r = requests.get('http://ergast.com/api/f1/{}.json'.format(self.season))
        j = json.loads(r.content.decode('utf-8'))
        rounds = pd.json_normalize(j['MRData']['RaceTable']['Races'])
        rounds = rounds[['season', 'round', 'raceName', 'date', 'time', 'Circuit.circuitId', 'Circuit.circuitName',
                         'Circuit.Location.lat', 'Circuit.Location.long', 'Circuit.Location.locality',
                         'Circuit.Location.country']]
        rounds.columns = ['season', 'round', 'raceName', 'date', 'time', 'circuitId', 'circuitName',
                          'lat', 'long', 'locality', 'country']
        rounds['round'] = rounds['round'].astype(int)
        self.events = rounds
        return rounds

    def get_drivers(self) -> pd.DataFrame:
        """
        Return a list of all drivers active in the given season

        Columns: ['number', 'code', 'givenName', 'familyName',
                  'dateOfBirth', 'nationality']
        Returns:
            Dataframe with drivers
        """
        if not self.drivers.empty:
            return self.drivers
        if self.season < 1950 or self.season > 2030:
            return None
        r = requests.get('http://ergast.com/api/f1/2021/drivers.json')
        j = json.loads(r.content.decode('utf-8'))
        drivers = pd.json_normalize(j['MRData']['DriverTable']['Drivers'])
        drivers = drivers[['permanentNumber', 'code', 'givenName', 'familyName',
                           'dateOfBirth', 'nationality']]
        drivers.columns = ['number', 'code', 'givenName', 'familyName',
                           'dateOfBirth', 'nationality']
        drivers['number'] = drivers['number'].astype(int)
        self.drivers = drivers
        return drivers

    def get_driver(self, did: str) -> pd.DataFrame:
        """
        Obtain the information of a driver based on its code, the three letter abbreviation
        used to identify the driver

        Columns: ['number', 'code', 'givenName', 'familyName',
                  'dateOfBirth', 'nationality']

        Args:
            did: Driver ID, e.g. 'ALO' for Fernando ALONSO

        Returns:
            Dataframe with one row with the information of the requested driver
            None, if driver not found
        """
        if isinstance(did, str):
            # Search by code and name
            res = self.drivers[self.drivers.code == did]
            if res.empty:
                res = self.drivers[self.drivers.familyName == did]
            return res
        elif isinstance(did, int):
            return self.drivers[self.drivers.number == did]
        else:
            return None

    @staticmethod
    def null_check(val: object) -> bool:
        """
        Checks if the given value is a Null value.
        Check is performaed for None, Null, NaT, NaN
        Args:
            val: Value to check

        Returns:
            False, in case of null object
            True, otherwise
        """
        if (not val) or pd.isnull(val) or pd.isna(val):
            return False
        return True

    def get_driver_fullname(self, driver: str) -> str:
        """
        Returns the fullname of a driver based on its driver code. First name in camel-case,
        lastname in uppercase. E.g., 'ALO' results in 'Fernando ALONSO'
        Args:
            driver: driver code

        Returns:
            Fullname of driver
        """
        if not self.null_check(driver):
            return ''
        driver = self.drivers[self.drivers.code == driver]
        return driver['givenName'].values[0] + ' ' + (driver['familyName'].values[0]).upper()

    def shorten_team_name(self, name: str) -> str:
        """
        Shorten a team name, more specific, remove ' team' at the end of the name
        Args:
            name: Fulname

        Returns:
            Shortened name
        """
        if not self.null_check(name):
            return ''
        if name.upper().endswith('TEAM'):
            return name[:-5]
        else:
            return name

    def set_default_event(self, grandprix: Union[str, int]) -> str:
        """
        Set the default event. If the parameter is a string, it is tested againt the name
        of the event, the country and the city the event takes place. In case of a number, this
        is the sequence number durin the year. It is also made the default event for
        following metohd calls.
        Args:
            grandprix: string or int identifying requested event

        Returns:
            The name of the event, or None if not found
        """
        df = self.get_events()
        if isinstance(grandprix, str):
            self.event = df[df.raceName.str.contains('(?i)' + grandprix)]['raceName']
            if not self.event.empty:
                self.event = self.event.values[0]
            else:
                self.event = df[df.country.str.contains('(?i)' + grandprix)]['raceName']
                if not self.event.empty:
                    self.event = self.event.values[0]
                else:
                    self.event = df[df.locality.str.contains('(?i)' + grandprix)]['raceName']
                    if not self.event.empty:
                        self.event = self.event.values[0]
                    else:
                        self.event = None
        elif isinstance(grandprix, int):
            self.event = df[df['round'] == grandprix]['raceName'].values[0]
        return self.event

    def get_session(self, grandprix: str = None, session: str = 'R') -> Laps:
        """
        Obtain the laps recorder for a single session of a grnadprix. If grandprix is not
        specified, the default event is used. If event or session is not cached, it is downloaded
        and cached.

        Args:
            grandprix: Event for which to return a session, if None default is used
            session: Session to return. Valid options are "R", "Q", "FP1", "FP2", FP3

        Returns:
            Laps dataframe object with the laps of the session
        """
        if not grandprix:
            grandprix = self.event
        if self.events.empty:
            _ = self.get_events()
        if self.drivers.empty:
            _ = self.get_drivers()
        if isinstance(grandprix, str) and isinstance(session, str):
            if grandprix in self.events.raceName.values:
                if grandprix in self.sessions:
                    return self.sessions[grandprix][session]
                self.sessions[grandprix] = {}
                self.sessions[grandprix]['R'] = ff1.get_session(self.season, grandprix).get_race().load_laps()
                self.sessions[grandprix]['Q'] = ff1.get_session(self.season, grandprix).get_quali().load_laps()
                self.sessions[grandprix]['FP1'] = ff1.get_session(self.season, grandprix).get_practice(1).load_laps()
                self.sessions[grandprix]['FP2'] = ff1.get_session(self.season, grandprix).get_practice(2).load_laps()
                self.sessions[grandprix]['FP3'] = ff1.get_session(self.season, grandprix).get_practice(3).load_laps()
                return self.sessions[grandprix][session]
        return None

    def get_qualify_laps(self, grandprix: str = None) -> Laps:
        """
        Return a dataframe of the qualification times per driver. For each qualifying session (Q1, Q2, Q3)
        the best lap per driver is listed.
        The start and end time of each qualifying session is determined on the Time field of the Lap. This
        field specifies the time since the start of datarecording. This is not a fixed moment, but approx 20
        minutes before the session starts.
        First, all minutes since the start of the session are grouped together when the time of the session
        start. A new group is created when there has been no lap completed for 5 minutes or more.
        If this results in more than three sessions, groups are combined based upon the number of drivers in
        the session. If a second time group has 18 cars running, it must be part of Q1 since Q2 only has
        15 competing cars. If there are still more than three groups, groups are combined based upon their
        duraion. Short sessions are combined when the start of the first and the finish of the second are within
        the duration of the qualifying session.

        Columns: ['Time', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime', 'Sector1Time',
                  'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
                  'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Compound', 'TyreLife', 'FreshTyre', 'LapStartTime',
                  'Team', 'Driver', 'TrackStatus', 'IsAccurate', 'SecFromStart', 'Q']
        Args:
            grandprix: Event for which to obtain the qualifying laps, if None, default event is used

        Returns:
            Dataframe with best laps per qualifying session
        """
        laps = self.get_session(grandprix, session='Q')
        times = laps['SecFromStart'] = laps['Time'].apply(lambda x: (math.floor(x.seconds / 60)))

        def split(lst, delta):
            def group(lst_to_group, max_delta):
                first = last = lst_to_group[0]
                for n in lst_to_group[1:]:
                    if last <= n - 1 <= last + max_delta:
                        last = n
                    else:  # Not part of the group, yield current group and start a new
                        yield first, last
                        first = last = n
                yield first, last  # Yield the last group

            dflist = []
            for begin, end in list(group(lst, delta)):
                nrdrivers = len(
                    laps[(begin <= laps.SecFromStart) & (laps.SecFromStart <= end)]['DriverNumber'].drop_duplicates())
                dflist.append([begin, end, nrdrivers])
            return pd.DataFrame(dflist, columns=['begin', 'end', 'noDrivers'])

        # split minutes from begin on empty periods of 5 minutes
        # most of the times this results in three groups for three Q's
        groups = split(times.sort_values().drop_duplicates().values, 2)

        # If there are more than 3 groups, combine where necessary
        # combine from the front in case of non fitting numer of particpants in a session
        if len(groups) > 3:
            participants = [0, 20, 15, 10, 0]
            q = 1
            i = 0
            while i < len(groups) - 1:
                if groups.iloc[i + 1]['noDrivers'] > participants[q + 1]:
                    groups.iloc[i]['end'] = groups.iloc[i + 1]['end']
                    groups.iloc[i]['noDrivers'] = max(groups.iloc[i]['noDrivers'], groups.iloc[i + 1]['noDrivers'])
                    groups = groups.drop([i + 1]).reset_index(drop=True)
                else:
                    i = i + 1
                    q = q + 1

        # First combine from the end of the list
        if len(groups) > 3:
            q = 3
            particapants = [0, 20, 15, 10]
            duration = [0, 20, 15, 10]
            for i in range(len(groups) - 1, 0, -1):
                print(i)
                if (groups.iloc[i - 1]['noDrivers'] <= particapants[q]) & \
                        (groups.iloc[i]['end'] - duration[q] <= groups.iloc[i - 1]['end']):
                    print('Too short session')
                    groups.iloc[i - 1]['end'] = groups.iloc[i]['end']
                    groups.iloc[i - 1]['noDrivers'] = max(groups.iloc[i - 1]['noDrivers'], groups.iloc[i]['noDrivers'])
                    groups = groups.drop([i])
                    q = q - 1
            print(groups)
        groups['Q'] = ['Q1', 'Q2', 'Q3']

        # Added Q session to each lap
        lookup = {}
        for _, row in groups.iterrows():
            for i in range(row['begin'], row['end'] + 1):
                lookup[i] = row['Q']
        laps['Q'] = laps.apply(lambda x: lookup[x.SecFromStart], axis=1)

        return laps

    def get_qualify_123_results(self, grandprix: str = None) -> pd.DataFrame:
        """
        Returns the results of the qualifying session. For each session a driver competed, his best time is
        listed, together with the compound used to set this time. The top 10 drivers have a time set for all
        three sessions, the bottom 5 for only the first session.
        Args:
            grandprix: Event for which the results are requested, if None, default event is used

        Colomns: ['DriverNumber', 'Driver', 'Team', 'Q3', 'Q3_C', 'Q2', 'Q2_C', 'Q1', 'Q1_C']

        Returns:
            Dataframe with qualifying results
        """
        laps = self.get_qualify_laps(grandprix)
        list_fastest_laps = list()
        for q in ['Q1', 'Q2', 'Q3']:
            _laps = laps[laps.Q == q]
            drivers = pd.unique(_laps['Driver'])
            for drv in drivers:
                drvs_fastest_lap = _laps.pick_driver(drv).pick_fastest()
                list_fastest_laps.append(drvs_fastest_lap)
        fastest_laps = Laps(list_fastest_laps).sort_values(by=['Q', 'LapTime']).reset_index(drop=True)
        pole_lap = fastest_laps.pick_fastest()
        fastest_laps['Delta'] = fastest_laps['LapTime'] - pole_lap['LapTime']
        fastest_laps['Gap'] = fastest_laps['LapTime'] - fastest_laps['LapTime'].shift(+1)

        Q1_laps = fastest_laps[fastest_laps.Q == 'Q1'][['DriverNumber', 'Driver', 'Team', 'LapTime', 'Compound']]
        Q1_laps = Q1_laps.sort_values('LapTime').reset_index(drop=True)
        Q2_laps = fastest_laps[fastest_laps.Q == 'Q2'][['DriverNumber', 'Driver', 'Team', 'LapTime', 'Compound']]
        Q2_laps = Q2_laps.sort_values('LapTime').reset_index(drop=True)
        Q3_laps = fastest_laps[fastest_laps.Q == 'Q3'][['DriverNumber', 'Driver', 'Team', 'LapTime', 'Compound']]
        Q3_laps = Q3_laps.sort_values('LapTime').reset_index(drop=True)

        drivers = pd.unique(fastest_laps['Driver'])
        drivers_out_Q1 = Q1_laps.iloc[15:, :]['Driver'].values
        drivers_to_Q2 = np.array([ele for ele in drivers if ele not in drivers_out_Q1])
        drivers_Q2 = Q2_laps.iloc[:, :]['Driver'].values
        drivers_not_driven_Q2 = np.array([ele for ele in drivers_to_Q2 if ele not in drivers_Q2])
        drivers_out_Q2 = Q2_laps.iloc[10:, :]['Driver'].values
        drivers_to_Q3 = Q2_laps.iloc[:10, :]['Driver'].values
        drivers_Q3 = Q3_laps.iloc[:, :]['Driver'].values
        drivers_not_driven_Q3 = np.array([ele for ele in drivers_to_Q3 if ele not in drivers_Q3])
        driver_results = (drivers_Q3.tolist() + drivers_not_driven_Q3.tolist() + drivers_out_Q2.tolist() +
                          drivers_not_driven_Q2.tolist() + drivers_out_Q1.tolist())

        Q_results = pd.DataFrame()
        for driver in driver_results:
            info = fastest_laps[fastest_laps.Driver == driver][['DriverNumber', 'Driver', 'Team']].drop_duplicates()
            if len(Q3_laps[Q3_laps.Driver == driver][['LapTime']]) > 0:
                info['Q3'] = Q3_laps[Q3_laps.Driver == driver][['LapTime']].values[0]
                info['Q3_C'] = Q3_laps[Q3_laps.Driver == driver][['Compound']].values[0]
            if len(Q2_laps[Q2_laps.Driver == driver][['LapTime']]) > 0:
                info['Q2'] = Q2_laps[Q2_laps.Driver == driver][['LapTime']].values[0]
                info['Q2_C'] = Q2_laps[Q2_laps.Driver == driver][['Compound']].values[0]
            info['Q1'] = Q1_laps[Q1_laps.Driver == driver][['LapTime']].values[0]
            info['Q1_C'] = Q1_laps[Q1_laps.Driver == driver][['Compound']].values[0]

            Q_results = Q_results.append(info)
        return Q_results.reset_index(drop=True)

    @staticmethod
    def compound_to_str(compound: str, single: bool = True, brackets: bool = True) -> str:
        """
        Convert compound to requested from. Either full name or only one character. Bracket are optional
        in both cases.
        Args:
            compound: String with full compund nam (e.g. 'SOFT')
            single: Return single character compund or full name
            brackets: Return compund name surrounded with brackets, or without

        Returns:
            Compound name
        """
        if isinstance(compound, str):
            if single:
                res = compound[:1] if len(compound) > 2 else ''
            else:
                res = compound if len(compound) > 2 else ''
            if brackets and len(res) >= 1:
                res = '(' + res + ')'
            return res
        else:
            return ''

    def print_qualify_123_results(self, grandprix: str = None) -> None:
        """
        Print the results of the qualification for a specific event
        Args:
            grandprix: Event to print

        Returns:
            None
        """
        result = self.get_qualify_123_results(grandprix)

        print('{:3}  {:>2}  {:<20}  {:<13}  {:<8} {:<3}  {:<8} {:<3}  {:<8} {:<3}'.
              format('Pos', 'Nr', 'Driver', 'Team', 'Q3', ' T ', 'Q2', ' T ', 'Q1', ' T '))
        for idx, row in result.iterrows():
            print('{:3d}  {:>2}  {:<20}  {:<13}  {:<8} {:<3}  {:<8} {:<3}  {:<8} {:<3}'.
                  format(idx + 1,
                         row['DriverNumber'],
                         self.get_driver_fullname(row['Driver']),
                         self.shorten_team_name(row['Team']),
                         self.timedelta_to_str(row['Q3']),
                         self.compound_to_str(row['Q3_C']),
                         self.timedelta_to_str(row['Q2']),
                         self.compound_to_str(row['Q2_C']),
                         self.timedelta_to_str(row['Q1']),
                         self.compound_to_str(row['Q1_C']),
                         ))

    def get_qualify_results(self, grandprix: str = None) -> pd.DataFrame:
        """
        Returns the results of the qualifying session. This implementation assumes one session qualidication
        so fastest time for each driver is taken.
        Dataset is extended with gap and interval data
        Args:
            grandprix: Event for which the results are requested, if None, default event is used

        Colomns: ['Time', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime', 'Sector1Time',
        'Sector2Time', 'Sector3Time', Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'SpeedI1',
        'SpeedI2', 'SpeedFL', 'SpeedST', 'Compound', 'TyreLife', 'FreshTyre', 'LapStartTime', 'Team', 'Driver',
        'TrackStatus', 'IsAccurate', 'Gap', 'Interval']

        Returns:
            Dataframe with qualifying results
        """

        laps = self.get_session(grandprix, session='Q')
        list_fastest_laps = list()
        drivers = pd.unique(laps['Driver'])
        for drv in drivers:
            drvs_fastest_lap = laps.pick_driver(drv).pick_fastest()
            list_fastest_laps.append(drvs_fastest_lap)
        fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)
        pole_lap = fastest_laps.pick_fastest()
        fastest_laps['Interval'] = fastest_laps['LapTime'] - pole_lap['LapTime']
        fastest_laps['Gap'] = fastest_laps['LapTime'] - fastest_laps['LapTime'].shift(+1)
        return fastest_laps

    def print_qualify_results(self, grandprix: str = None) -> None:
        """
        Print qualification results for a session of the given grandprix (or default in case None is passed)
        Assumes qualification of one run.
        Format:
        Pos  Nr  Driver                Team           Tyre Time       Interval   Gap
          1  33  Max VERSTAPPEN        Red Bull       S    1:03.720
          2   4  Lando NORRIS          McLaren        S    1:03.768   0:00.048   0:00.048
          3  11  Sergio PÉREZ          Red Bull       S    1:03.990   0:00.270   0:00.222

        Args:
            grandprix: Event for which the qualification results must be printed

        Returns:
            None
        """
        result = self.get_qualify_results(grandprix)
        print('{:3}  {:2}  {:<20}  {:<13}  {:<4} {:<9}  {:<9}  {:<9}'.
              format('Pos', 'Nr', 'Driver', 'Team', 'Tyre', 'Time', 'Interval', 'Gap'))
        for idx, row in result.iterrows():
            print('{:3d}  {:>2}  {:<20}  {:<13}  {:<4} {:<9}  {:<9}  {:<9}'.
                  format(idx + 1,
                         row['DriverNumber'],
                         self.get_driver_fullname(row['Driver']),
                         self.shorten_team_name(row['Team']),
                         row['Compound'][:1],
                         self.timedelta_to_str(row['LapTime']),
                         self.timedelta_to_str(row['Interval']),
                         self.timedelta_to_str(row['Gap'])
                         ))

    @staticmethod
    def timedelta_to_str(td: timedelta) -> str:
        """
        Conver a timedelta to a string in the form MM:ss:mmm
        Args:
            td: timedelta to convert

        Returns:
            string wiith timedelta representation
        """
        if pd.isnull(td):
            return ''
        if isinstance(td, str):
            return ''
        seconds = td.seconds
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60
        millsec = math.floor(td.microseconds / 1000)
        if minutes == seconds == millsec == 0:
            return ''
        return '{:1d}:{:02d}.{:03d}'.format(minutes, seconds, millsec)

    @staticmethod
    def get_diff_str(time1: timedelta, time2: timedelta) -> str:
        """
        Calculate the difference between the times and return it as a string in the form '- 0.313'.
        The sign will be negative if the second time is smaller than the first.
        Args:
            time1: First time
            time2: Second time

        Returns:
            String with the time difference
        """
        if time1 >= time2:
            delta = (time1 - time2).total_seconds()
            sign = '-'
        else:
            delta = (time2 - time1).total_seconds()
            sign = '+'
        return '{}{:2d}.{:03d}'.format(sign, int(math.floor(delta)), int(1000 * (delta % 1)))

    def compare_laps(self, lap1: Lap, lap2: Lap) -> str:
        """
        Compare two laps and print an overview of the comaprison, including sector time comparison
        Args:
            lap1: First lap
            lap2: Second lap

        Returns:
            The string with the comparison
        """
        d1 = self.get_driver(lap1['Driver'])
        d2 = self.get_driver(lap2['Driver'])

        format_string = '{:<14}| {:>18} |{:>14}'

        left = d1['familyName'].values[0].upper()
        rght = d2['familyName'].values[0].upper()
        diff = '{}{:>10}'.format('sector 1', self.get_diff_str(lap1['Sector1Time'], lap2['Sector1Time']))
        l1 = format_string.format(left, diff, rght)

        left = '{:2d} - {}'.format(d1['number'].values[0], d1['code'].values[0])
        rght = '{:2d} - {}'.format(d2['number'].values[0], d2['code'].values[0])
        diff = '{}{:>10}'.format('sector 2', self.get_diff_str(lap1['Sector2Time'], lap2['Sector2Time']))
        l2 = format_string.format(left, diff, rght)

        left = self.shorten_team_name(lap1['Team'].upper())
        rght = self.shorten_team_name(lap2['Team'].upper())
        diff = '{}{:>10}'.format('sector 3', self.get_diff_str(lap1['Sector3Time'], lap2['Sector3Time']))
        l3 = format_string.format(left, diff, rght)

        left = '{} ({:2.0f})'.format(lap1['Compound'], (lap1['TyreLife']))
        rght = '{} ({:2.0f})'.format(lap2['Compound'], (lap2['TyreLife']))
        diff = ''
        l4 = format_string.format(left, diff, rght)

        left = self.timedelta_to_str(lap1['LapTime'])
        rght = self.timedelta_to_str(lap2['LapTime'])
        diff = '{:<8}{:>10}'.format('lap', self.get_diff_str(lap1['LapTime'], lap2['LapTime']))
        l5 = format_string.format(left, diff, rght)

        result = l1 + '\n' + l2 + '\n' + l3 + '\n' + l4 + '\n' + l5 + '\n'
        print(result)
        return result

    def print_laptime_table(self, laptimetable: pd.DataFrame) -> None:
        """
        Print a table with laptimes in the form:
        Pos  Nr  Driver                Team           Lap time   Sector 1   Sector 2   Sector 3
          2  33  Max VERSTAPPEN        Red Bull       1:03.690   0:16.203   0:28.250   0:19.237
        ...
        Position is the index of the entry, increased with one.
        Args:
            laptimetable: Table with laptimes to print

        Returns:
            None
        """
        print('{:3}  {:>2}  {:<20}  {:<13}  {:<9}  {:<9}  {:<9}  {:<9}'.
              format('Pos', 'Nr', 'Driver', 'Team', 'Lap time', 'Sector 1', 'Sector 2', 'Sector 3'))
        for idx, row in laptimetable.iterrows():
            print('{:3d}  {:>2}  {:<20}  {:<13}  {:<9}  {:<9}  {:<9}  {:<9}'.
                  format(idx + 1,
                         row['DriverNumber'],
                         self.get_driver_fullname(row['Driver']),
                         self.shorten_team_name(row['Team']),
                         self.timedelta_to_str(row['LapTime']),
                         self.timedelta_to_str(row['Sector1Time']),
                         self.timedelta_to_str(row['Sector2Time']),
                         self.timedelta_to_str(row['Sector3Time'])
                         ))

    @staticmethod
    def determine_best_combined_laptimes(laps: pd.DataFrame) -> pd.DataFrame:
        """
        Determine the best possible laptim per driver by combing the best sector times.
        Also determine the best possible time by combining the fastest sector times over
        all drivers. Add driver with number 00 with this laptime
        Args:
            laps: Set of laptimes, e.g. a sesion

        Returns:
            DataFrame with best possible laptimes per driver, including overall best
            possible time
        """
        flaps = laps[['DriverNumber', 'Sector1Time', 'Sector2Time', 'Sector3Time']]
        flaps = flaps.groupby('DriverNumber').min().reset_index()
        flaps = flaps.append({"DriverNumber": 0,
                              "Sector1Time": laps.Sector1Time.min(),
                              "Sector2Time": laps.Sector2Time.min(),
                              "Sector3Time": laps.Sector3Time.min()},
                             ignore_index=True)
        flaps['LapTime'] = flaps['Sector1Time'] + flaps['Sector2Time'] + flaps['Sector3Time']
        flaps = flaps.sort_values('LapTime')
        flaps = flaps.merge(laps[['DriverNumber', 'Driver', 'Team']].drop_duplicates(), how='left')
        return flaps

    def add_text(self, canvas: ImageDraw, x: int, y: int, text: str, color: ImageColor,
                 font: ImageFont, align: str = 'left') -> None:
        """
        Add the specified text to the image. The text is added add lcation (x,y). The text is located right from
        this point in case of right alignement and left in case of left alignment. The font and color to use to
        draw the text are passed as parameter.
        Args:
            canvas:
            x: x location to add text
            y: y locatio to add text
            text: The text to add to the image
            color: The color to paint the text with
            font: THe font to use to print the text
            align: Alignment, 'left' or 'right'

        Returns:
            None
        """
        if align == 'right':
            w, h = canvas.textsize(text, font=font)
            x = x - w
        canvas.text((x, y), text, color, font=font)

    def add_time_diff(self, canvas: ImageDraw, x: int, y: int, text: str, font: ImageFont, align: str = 'left') -> None:
        """
        Add the time difference string to the draing canvas. Yellow when it is a negative difference, green otherwise.
        See als draw_text, as this method only change color depending the color to display the time difference.
        Args:
            canvas: ImageDraw to add text to
            x: x-location
            y: y-location
            text: Text to add
            font: Font to use
            align: 'left' or 'right'

        Returns:
            None
        """
        color = ImageColor.getrgb("green") if text.strip()[:1] == '-' else ImageColor.getrgb("yellow")
        self.add_text(canvas, x, y, text, color, font, align)

    def compare_laps_graphic(self, lap1: Lap, lap2: Lap, show: bool = True) -> Image:
        """
        Show a PNG with a comparion of two laptimes, including sector compare
        Args:
            lap1: First lap, shown left
            lap2: Second lap, shown right
            show: If True, ask OS to show the generated image
        Returns:
            Image object with the created image
        """
        width = 700
        height = 200
        hmargin = 10
        vstart = 20
        vspacing = 30

        font1 = ImageFont.truetype("fonts\\Formula1-Regular.otf", 20)
        font2 = ImageFont.truetype("fonts\\Formula1-Bold.otf", 22)
        font3 = ImageFont.truetype("fonts\\Formula1-Wide.otf", 16)

        white = ImageColor.getrgb("white")
        grey = ImageColor.getrgb("grey")

        img = Image.new(mode="RGB", size=(width, height))
        draw = ImageDraw.Draw(img)

        d1 = self.get_driver(lap1['Driver'])
        d2 = self.get_driver(lap2['Driver'])

        # Lap left
        line = d1['familyName'].values[0].upper()
        self.add_text(draw, hmargin, vstart + 0 * vspacing, line, white, font2, align='left')
        line = '{:2d} - {}'.format(d1['number'].values[0], d1['code'].values[0])
        self.add_text(draw, hmargin, vstart + 1 * vspacing, line, white, font1, align='left')
        line = self.shorten_team_name(lap1['Team'].upper())
        self.add_text(draw, hmargin, vstart + 2 * vspacing, line, white, font1, align='left')
        line = '{} ({:2.0f})'.format(lap1['Compound'], (lap1['TyreLife']))
        self.add_text(draw, hmargin, vstart + 3 * vspacing, line, white, font1, align='left')
        line = self.timedelta_to_str(lap1['LapTime'])
        self.add_text(draw, hmargin, vstart + 5 * vspacing, line, white, font3, align='left')

        # Seperators
        draw.line([200, 0, 200, height], fill=grey, width=2)
        draw.line([width - 200, 0, width - 200, height], fill=grey, width=2)

        # Lap Right
        line = d2['familyName'].values[0].upper()
        self.add_text(draw, width - hmargin, vstart + 0 * vspacing, line, white, font2, align='right')
        line = '{:2d} - {}'.format(d2['number'].values[0], d2['code'].values[0])
        self.add_text(draw, width - hmargin, vstart + 1 * vspacing, line, white, font1, align='right')
        line = self.shorten_team_name(lap2['Team'].upper())
        self.add_text(draw, width - hmargin, vstart + 2 * vspacing, line, white, font1, align='right')
        line = '{} ({:2.0f})'.format(lap2['Compound'], (lap2['TyreLife']))
        self.add_text(draw, width - hmargin, vstart + 3 * vspacing, line, white, font1, align='right')
        line = self.timedelta_to_str(lap2['LapTime'])
        self.add_text(draw, width - hmargin, vstart + 5 * vspacing, line, white, font3, align='right')

        # Comparison static
        self.add_text(draw, 220, vstart + 1 * vspacing, 'sector 1', white, font1, align='left')
        self.add_text(draw, 220, vstart + 2 * vspacing, 'sector 2', white, font1, align='left')
        self.add_text(draw, 220, vstart + 3 * vspacing, 'sector 3', white, font1, align='left')
        self.add_text(draw, 220, vstart + 5 * vspacing, 'lap', white, font2, align='left')
        # Comparison dynamic
        diff = '{:>10}'.format(self.get_diff_str(lap1['Sector1Time'], lap2['Sector1Time']))
        self.add_time_diff(draw, width - 220, vstart + 1 * vspacing, diff, font2, align='right')
        diff = '{:>10}'.format(self.get_diff_str(lap1['Sector2Time'], lap2['Sector2Time']))
        self.add_time_diff(draw, width - 220, vstart + 2 * vspacing, diff, font2, align='right')
        diff = '{:>10}'.format(self.get_diff_str(lap1['Sector3Time'], lap2['Sector3Time']))
        self.add_time_diff(draw, width - 220, vstart + 3 * vspacing, diff, font2, align='right')
        diff = '{:>10}'.format(self.get_diff_str(lap1['LapTime'], lap2['LapTime']))
        self.add_time_diff(draw, width - 220, vstart + 5 * vspacing, diff, font2, align='right')

        # %matplotlib inline
        if show:
            imshow(np.asarray(img))
            img.show()
        return img

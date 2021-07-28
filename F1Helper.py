import fastf1 as ff1
import requests
import json
import pandas as pd
import numpy as np
import math
import os

from fastf1.core import Laps


class F1Helper:

    def __init__(self):
        self.season = 0
        self.events = pd.DataFrame()
        self.drivers = pd.DataFrame()
        self.sessions = {}
        self.event = None
        if not os.path.exists('cache'):
            os.makedirs('cache')
        ff1.Cache.enable_cache('cache')

    def set_season(self, year):
        if isinstance(year, int) and 1950 < year < 2022:
            new_season = year
        else:
            new_season = 2021
        if new_season != self.season:
            self.season = new_season
            self.events = pd.DataFrame()
            self.drivers = pd.DataFrame()
            self.sessions = {}

    def get_season(self):
        return self.season

    def get_events(self):
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

    def get_drivers(self):
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

    def get_driver(self, did):
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

    def null_check(self, val):
        if (not val) or pd.isnull(val) or pd.isna(val):
            return False
        return True

    def get_driver_fullname(self, driver):
        if not self.null_check(driver):
            return ''
        driver = self.drivers[self.drivers.code == driver]
        return driver['givenName'].values[0] + ' ' + (driver['familyName'].values[0]).upper()

    def shorten_team_name(self, name):
        if not self.null_check(name):
            return ''
        if name.upper().endswith('TEAM'):
            return name[:-5]
        else:
            return name

    def set_event(self, grandprix):
        if self.events.empty:
            _ = self.get_events()

    def set_default_event(self, grandprix):
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

    def get_session(self, grandprix=None, session='R'):
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

    def get_qualify_laps(self, grandprix=None):
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

    def get_qualify_123_results(self, grandprix=None):
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
        driver_results = drivers_Q3.tolist() + drivers_not_driven_Q3.tolist() + drivers_out_Q2.tolist() + \
                         drivers_not_driven_Q2.tolist() + drivers_out_Q1.tolist()

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

    def print_qualify_123_results(self, grandprix=None):
        result = self.get_qualify_123_results(grandprix).fillna(' ')

        def compound_to_str(compound, single=True, brackets=True):
            return '(' + compound[:1] + ')' if len(compound) > 2 else ''

        print('{:3}  {:>2}  {:<20}  {:<13}  {:<8} {:<3}  {:<8} {:<3}  {:<8} {:<3}'. \
              format('Pos', 'Nr', 'Driver', 'Team', 'Q3', ' T ', 'Q2', ' T ', 'Q1', ' T '))
        for idx, row in result.iterrows():
            print('{:3d}  {:>2}  {:<20}  {:<13}  {:<8} {:<3}  {:<8} {:<3}  {:<8} {:<3}'. \
                  format(idx + 1,
                         row['DriverNumber'],
                         self.get_driver_fullname(row['Driver']),
                         self.shorten_team_name(row['Team']),
                         self.timedelta_to_str(row['Q3']),
                         compound_to_str(row['Q3_C']),
                         self.timedelta_to_str(row['Q2']),
                         compound_to_str(row['Q2_C']),
                         self.timedelta_to_str(row['Q1']),
                         compound_to_str(row['Q1_C']),
                         ))

    def get_qualify_results(self, grandprix=None):
        laps = self.get_session(grandprix, session='Q')
        list_fastest_laps = list()
        drivers = pd.unique(laps['Driver'])
        for drv in drivers:
            drvs_fastest_lap = laps.pick_driver(drv).pick_fastest()
            list_fastest_laps.append(drvs_fastest_lap)
        fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)
        pole_lap = fastest_laps.pick_fastest()
        fastest_laps['Delta'] = fastest_laps['LapTime'] - pole_lap['LapTime']
        fastest_laps['Gap'] = fastest_laps['LapTime'] - fastest_laps['LapTime'].shift(+1)
        return fastest_laps

    def print_qualify_results(self, grandprix=None):
        result = self.get_qualify_results(grandprix)
        print('{:3}  {:2}  {:<20}  {:<13}  {:<4} {:<9}  {:<9}  {:<9}'.
              format('Pos', 'Nr', 'Driver', 'Team', 'Tyre', 'Time', 'Delta', 'Gap'))
        for idx, row in result.iterrows():
            print('{:3d}  {:>2}  {:<20}  {:<13}  {:<4} {:<9}  {:<9}  {:<9}'.
                  format(idx + 1,
                         row['DriverNumber'],
                         self.get_driver_fullname(row['Driver']),
                         self.shorten_team_name(row['Team']),
                         row['Compound'][:1],
                         self.timedelta_to_str(row['LapTime']),
                         self.timedelta_to_str(row['Delta']),
                         self.timedelta_to_str(row['Gap'])
                         ))

    def timedelta_to_str(self, td):
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

    def compare_laps(self, lap1, lap2):
        d1 = self.get_driver(lap1['Driver'])
        d2 = self.get_driver(lap2['Driver'])

        def get_diff_str(time1, time2):
            if time1 >= time2:
                delta = (time1 - time2).total_seconds()
                sign = '-'
            else:
                delta = (time2 - time1).total_seconds()
                sign = '+'
            return '{}{:2d}.{:03d}'.format(sign, int(math.floor(delta)), int(1000 * (delta % 1)))

        format_string = '{:<14}| {:>18} |{:>14}'

        left = d1['familyName'].values[0].upper()
        rght = d2['familyName'].values[0].upper()
        diff = '{}{:>10}'.format('sector 1', get_diff_str(lap1['Sector1Time'], lap2['Sector1Time']))
        l1 = format_string.format(left, diff, rght)

        left = '{:2d} - {}'.format(d1['number'].values[0], d1['code'].values[0])
        rght = '{:2d} - {}'.format(d2['number'].values[0], d2['code'].values[0])
        diff = '{}{:>10}'.format('sector 2', get_diff_str(lap1['Sector2Time'], lap2['Sector2Time']))
        l2 = format_string.format(left, diff, rght)

        left = self.shorten_team_name(lap1['Team'].upper())
        rght = self.shorten_team_name(lap2['Team'].upper())
        diff = '{}{:>10}'.format('sector 3', get_diff_str(lap1['Sector3Time'], lap2['Sector3Time']))
        l3 = format_string.format(left, diff, rght)

        left = '{} ({:2.0f})'.format(lap1['Compound'], (lap1['TyreLife']))
        rght = '{} ({:2.0f})'.format(lap2['Compound'], (lap2['TyreLife']))
        diff = ''
        l4 = format_string.format(left, diff, rght)

        left = self.timedelta_to_str(lap1['LapTime'])
        rght = self.timedelta_to_str(lap2['LapTime'])
        diff = '{:<8}{:>10}'.format('lap', get_diff_str(lap1['LapTime'], lap2['LapTime']))
        l5 = format_string.format(left, diff, rght)

        print(l1)
        print(l2)
        print(l3)
        print(l4)
        print(l5)

    def print_laptime_table(self, laptimetable):
        result = laptimetable
        print('{:3}  {:>2}  {:<20}  {:<13}  {:<9}  {:<9}  {:<9}  {:<9}'. \
              format('Pos', 'Nr', 'Driver', 'Team', 'Lap time', 'Sector 1', 'Sector 2', 'Sector 3'))
        for idx, row in result.iterrows():
            print('{:3d}  {:>2}  {:<20}  {:<13}  {:<9}  {:<9}  {:<9}  {:<9}'. \
                  format(idx + 1,
                         row['DriverNumber'],
                         self.get_driver_fullname(row['Driver']),
                         self.shorten_team_name(row['Team']),
                         self.timedelta_to_str(row['LapTime']),
                         self.timedelta_to_str(row['Sector1Time']),
                         self.timedelta_to_str(row['Sector2Time']),
                         self.timedelta_to_str(row['Sector3Time'])
                         ))

    def determine_best_combined_laptimes(self, laps):
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

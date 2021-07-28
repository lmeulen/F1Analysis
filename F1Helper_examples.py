from F1Helper import F1Helper

print('Initialisaiton and first getters')
f1h = F1Helper()
f1h.set_season(2021)
print(f1h.get_season())
print(f1h.get_events().head(1)[['round', 'raceName', 'circuitName']].values)
print(f1h.get_drivers().head(1)[['number', 'code', 'givenName', 'familyName']].values)
print(f1h.get_session('Styrian Grand Prix', 'R').tail(1)[['DriverNumber', 'LapTime']].values)

print()
print('Comparing two laps:')
f1h.set_season(2021)
f1h.set_default_event('Aus')
laps = f1h.get_session(session='Q')
alo_lap = laps.pick_driver("ALO").pick_fastest()
vet_lap = laps.pick_driver("VET").pick_fastest()
f1h.compare_laps(alo_lap, vet_lap)

print()
print('Qualification results (fastest lap in all Q sessions):')
f1h.print_qualify_results()

print()
print('Qualification results (Q1/Q2/Q3):')
f1h.print_qualify_123_results()

print()
print('Best possible laptimes')
laps = f1h.get_session(session='Q')
flaps = f1h.determine_best_combined_laptimes(laps)
f1h.print_laptime_table(flaps)

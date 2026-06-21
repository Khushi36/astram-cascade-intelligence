
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta

file_path = r'c:\Users\Khushi\Downloads\Traffic\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'

rows = []
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f'Total rows: {len(rows)}')
causes = Counter(r['event_cause'] for r in rows)
print('Event causes:', dict(causes.most_common(15)))
priority = Counter(r['priority'] for r in rows)
print('Priority:', dict(priority))
closure = Counter(r['requires_road_closure'] for r in rows)
print('Road closure:', dict(closure))
status = Counter(r['status'] for r in rows)
print('Status:', dict(status))

IST = timedelta(hours=5, minutes=30)
resolution_times = []
for r in rows:
    try:
        start = r['start_datetime']
        closed = r['closed_datetime']
        if start and closed and closed not in ('NULL',''):
            s = datetime.fromisoformat(start.replace('+00', '+00:00').split('.')[0].rstrip('Z'))
            c = datetime.fromisoformat(closed.replace('+00', '+00:00').split('.')[0].rstrip('Z'))
            diff = (c - s).total_seconds() / 60
            if 0 < diff < 20000:
                resolution_times.append((r['event_cause'], diff, r['corridor'], r['priority']))
    except Exception as e:
        pass

by_cause = defaultdict(list)
for cause, t, corr, p in resolution_times:
    by_cause[cause].append(t)
print('\nAvg resolution time (minutes) by cause:')
for cause, times in sorted(by_cause.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
    avg = sum(times)/len(times)
    med = sorted(times)[len(times)//2]
    print(f'  {cause}: avg={avg:.0f}min median={med:.0f}min n={len(times)}')

corridors = Counter(r['corridor'] for r in rows)
print('\nTop 15 corridors:', dict(corridors.most_common(15)))

veh = Counter(r['veh_type'] for r in rows if r.get('veh_type','') not in ('','NULL','0',None))
print('\nVehicle types:', dict(veh.most_common(10)))

hour_counts = Counter()
dow_counts = Counter()
for r in rows:
    try:
        s = r['start_datetime']
        if s and s not in ('NULL',''):
            dt = datetime.fromisoformat(s.replace('+00', '+00:00').split('.')[0])
            ist_dt = dt + IST
            hour_counts[ist_dt.hour] += 1
            dow_counts[ist_dt.strftime('%A')] += 1
    except:
        pass
print('\nHourly counts IST:', dict(sorted(hour_counts.items())))
print('\nDay of week:', dict(dow_counts))

# corridor priority breakdown
corridor_priority = defaultdict(Counter)
for r in rows:
    if r['corridor'] not in ('NULL','','Non-corridor'):
        corridor_priority[r['corridor']][r['priority']] += 1
print('\nHigh priority pct per corridor (top 10):')
top_corr = [(c, dict(v)) for c,v in corridor_priority.items()]
top_corr.sort(key=lambda x: x[1].get('High',0), reverse=True)
for c, v in top_corr[:10]:
    total = sum(v.values())
    pct = v.get('High',0)/total*100
    h = v.get('High',0)
    print(f'  {c}: {h}/{total} = {pct:.0f}% High')

# event cascade: same corridor same day
print('\nCorridors with 5+ events in a single day (top):')
corr_day = defaultdict(list)
for r in rows:
    if r['corridor'] not in ('NULL','','Non-corridor'):
        try:
            s = r['start_datetime']
            dt = datetime.fromisoformat(s.replace('+00', '+00:00').split('.')[0])
            ist = dt + IST
            corr_day[(r['corridor'], ist.date())].append(r['event_cause'])
        except:
            pass
heavy = [(k,v) for k,v in corr_day.items() if len(v)>=5]
heavy.sort(key=lambda x: -len(x[1]))
for (corr,day), evts in heavy[:10]:
    print(f'  {corr} {day}: {len(evts)} events -> {Counter(evts).most_common(3)}')

# road closure = TRUE count
rc_true = [r for r in rows if r.get('requires_road_closure','').upper() == 'TRUE']
print(f'\nRoad closure required: {len(rc_true)} ({len(rc_true)/len(rows)*100:.1f}%)')
rc_causes = Counter(r['event_cause'] for r in rc_true)
print('Road closure by cause:', dict(rc_causes.most_common()))

# authenticated field
auth = Counter(r['authenticated'] for r in rows)
print('\nAuthenticated:', dict(auth))

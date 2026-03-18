#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

@dataclass
class ProfileEvent:
    smid: int
    cycle: int
    pc: int
    phase: int

# Read the profile.out file
sm_events = {}
current_smid = None
start_cycle = 0

def parse_part(part):
    key, value = part.split('=')
    return key.strip(), int(value.strip())

with open('profile.out', 'r') as f:
    for line in f:
        if "Profile data for SM" in line:
            current_smid = line.strip().split()[-1].rstrip(':')
            current_smid = int(current_smid)
        elif 'Timestamp' in line:
            parts = line.split(':')[1].strip()
            parts = [x.strip() for x in parts.split(',')]
            parts = {
                k: v for k, v in map(parse_part, parts)
            }
            event = ProfileEvent(
                smid=current_smid,
                cycle=parts['Timestamp'],
                pc=parts['PC'],
                phase=parts['Phase']
            )
            if current_smid not in sm_events:
                sm_events[current_smid] = []
            sm_events[current_smid].append(event)

# for now since it's not time-aligned, just plot per-SM timelines
def normalize_events(events):
    if not events:
        return events
    min_cycle = min(event.cycle for event in events)
    for event in events:
        event.cycle -= min_cycle
    return events

for event in sm_events.values():
    normalize_events(event)

phase_styles = {
    1: {'c': 'r', 'marker': 'o', 'label': 'IFU'},  # IFU phase
    2: {'c': 'g', 'marker': '^', 'label': 'DEP_LOAD'},  # DEP_LOAD phase
    3: {'c': 'black', 'marker': 'x', 'label': 'FINISH'},  # FINISH phase
    4: {'c': 'blue', 'marker': '^', 'label': 'TMA'},  # FINISH phase

    100: {'c': 'orange', 'marker': 'o', 'label': 'USER_0'},
    101: {'c': 'orange', 'marker': 'x', 'label': 'USER_1'},
    102: {'c': 'orange', 'marker': '^', 'label': 'USER_2'},
}

plt.figure(figsize=(40, 20))
frequency = 1.98
phase_grouped = {}
for smid, events in sm_events.items():
    for event in events:
        phase = event.phase
        if phase not in phase_grouped:
            phase_grouped[phase] = {'x': [], 'y': []}
        time_us = event.cycle / frequency / 1000
        phase_grouped[phase]['x'].append(time_us)
        phase_grouped[phase]['y'].append(smid)

for phase, data in phase_grouped.items():
    style = phase_styles.get(phase, {'c': 'k', 'marker': 'o'})
    plt.scatter(data['x'], data['y'], **style, alpha=0.6)

ticks = plt.gca().get_xticks()
plt.xticks(np.arange(0, max(ticks), 1))
plt.xlabel('Time (us)')
plt.ylabel('SM ID')
plt.title('DAE Profile Events per SM')
plt.legend()
plt.grid(True)
plt.savefig('dae_profile.png')

from config import ANCHORS, target, walls
from simulator import simulate_rssi
from geometry import count_walls
from localization import rssi_to_distance_wall_aware
from multilateration import estimate_position, position_error
import numpy as np


n_samples = 15

rssi_samples = {}
mean_rssi = {}
variance_rssi = {}

# Collect multiple noisy RSSI samples for each anchor
for name, anchor in ANCHORS.items():
    samples = []

    for i in range(n_samples):
        rssi = simulate_rssi(anchor, target, walls)
        samples.append(rssi)

    rssi_samples[name] = samples
    mean_rssi[name] = np.mean(samples)
    variance_rssi[name] = np.var(samples)


# Convert mean RSSI into one wall-aware distance per anchor
distances = {}

for name, anchor in ANCHORS.items():
    wall_count = count_walls(anchor, target, walls)

    distances[name] = rssi_to_distance_wall_aware(
        mean_rssi[name],
        wall_count
    )

# Build distances using only ONE RSSI sample per anchor
single_sample_distances = {}

for name, anchor in ANCHORS.items():
    wall_count = count_walls(anchor, target, walls)

    single_rssi = rssi_samples[name][0]

    single_sample_distances[name] = rssi_to_distance_wall_aware(
        single_rssi,
        wall_count
    )

# Position using mean/filtered RSSI
filtered_estimated = estimate_position(ANCHORS, distances)
filtered_error = position_error(target, filtered_estimated)


# Position using only one RSSI reading
single_estimated = estimate_position(ANCHORS, single_sample_distances)
single_error = position_error(target, single_estimated)

# Print each anchor
for name in ANCHORS:
    print(name)
    print("Mean RSSI:", mean_rssi[name])
    print("RSSI variance:", variance_rssi[name])
    print("Estimated distance:", distances[name])
    print()


print("Actual position:", target)

print("\nSingle-sample position:", single_estimated)
print("Single-sample error:", single_error)

print("\nMean-RSSI position:", filtered_estimated)
print("Mean-RSSI error:", filtered_error)
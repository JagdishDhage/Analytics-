# Map POI tables

A POI table names real places on a map so reports can say "dropped Pochinki"
instead of "dropped D5". It is calibration data, like a profile: coordinates
depend on the map's world size and have to be measured, not guessed. Nothing
here ships with invented coordinates for real BGMI locations.

Format:

```json
{
  "map": "ERANGEL",
  "world_size_m": 8000,
  "pois": [
    {"name": "POCHINKI", "x_m": 4100.0, "y_m": 4200.0, "radius_m": 250.0}
  ]
}
```

To build one, open a frame of the map stream and click the places you care
about:

```bash
python -m bgmi_scrim mapcalibrate stream.mp4 --profile profiles/myorg_map.json \
    --pois maps/erangel.json --at 90
```

Each click asks for a name and writes a POI at the world coordinate under the
cursor. Without a POI table, every report falls back to grid references
(`A1`..`H8` by default), which need no data and are still precise enough to
talk about a rotation.

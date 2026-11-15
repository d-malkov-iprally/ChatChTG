"""The classes of the detector and the CubiCasa5K names that they come from."""

from __future__ import annotations

# The index of a class is its category id in the dataset and in the model.
CLASSES = ["window", "door", "toilet", "sink", "bathtub", "shower", "sauna bench"]

# The second word of a fixture's class in model.svg ("FixedFurniture Toilet") -> our class.
# A name that is not here is not a class of the detector.
FIXTURES = {
    # A urinal is not a WC seat, so it is not a toilet.
    "Toilet": "toilet",
    # A double sink is one fixture, and the benchmark counts it as one sink.
    "Sink": "sink",
    "SideSink": "sink",
    "RoundSink": "sink",
    "CornerSink": "sink",
    "DoubleSink": "sink",
    "DoubleSinkRight": "sink",
    "Bathtub": "bathtub",
    "BathtubRound": "bathtub",
    "Jacuzzi": "bathtub",
    # A shower screen and a shower cabin are drawn around a shower. They are a part of it,
    # not a second shower: with ShowerCab as a shower, 178 plans have one shower more than
    # the benchmark's answer.
    "Shower": "shower",
    "ShowerPlatform": "shower",
    "SaunaBench": "sauna bench",
    "SaunaBenchHigh": "sauna bench",
    "SaunaBenchLow": "sauna bench",
    "SaunaBenchMid": "sauna bench",
}

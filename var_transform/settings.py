from dataclasses import dataclass
from search import *

@dataclass
class NoisyVar:
    lower: int
    upper: int
    sampling_num: int

@dataclass
class Search:
    way: str
    threshold: float

@dataclass
class Settings:
    algorithm: str
    noisy_var: NoisyVar
    integral: str
    search: Search

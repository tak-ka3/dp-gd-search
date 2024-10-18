from dataclasses import dataclass
from search import *

@dataclass
class Input:
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
    input: Input
    integral: str
    search: Search

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

class Config:
    def __init__(self, settings: Settings):
        self.input = settings.input
        self.integral = settings.integral
        self.algorithm = settings.algorithm
        self.saerch = settings.search

    def _read_search(search):
        if search.way == "all":
            return search_all
        elif search.way == "threshold":
            return search_by_threshold

from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    name = "base"

    def __init__(self, region):
        self.region = region

    @abstractmethod
    def fetch(self, start, end, **kwargs):
        raise NotImplementedError

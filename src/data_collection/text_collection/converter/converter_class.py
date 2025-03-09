from pathlib import Path
from abc import ABC, abstractmethod

class Converter(ABC):

    def __init__(self, raw_files_dir: Path, prepared_files_dir: Path):
        self.raw_files_dir = raw_files_dir
        self.prepared_files_dir = prepared_files_dir
   
    @abstractmethod
    def extract_text():
        pass
    
    @abstractmethod
    def save_text():
        pass

    @abstractmethod
    def convert_files(self):
        pass
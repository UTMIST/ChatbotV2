from abc import ABC, abstractmethod
from pandas import DataFrame
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DataSourceProcessStatus(Enum):

    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class DataSourceConfig:
    """
    This class should contain the configuration for the data source.
    """
    pass


@dataclass
class DataTransformConfig:
    """
    This class should contain the configuration for the data transformation process.
    """
    pass


@dataclass
class DataLoadConfig:
    """
    This class should contain the configuration for the data loading process.
    """
    pass


class DataSource(ABC):

    def __init__(self, config: DataSourceConfig):
        self.config: DataSourceConfig = config
        self.data : Optional[DataFrame] = None  

    @abstractmethod
    def extract_data(self) -> None:
        """
        This method should load the data from the data source into the data warehouse.
        """
        pass

    @abstractmethod
    def get_raw_data(self) -> DataFrame:
        """
        This method should return a DataFrame with the raw data from the data source.
        """

    @abstractmethod
    def update_process_status(self, status: DataSourceProcessStatus) -> None:
        """
        This method should update the process status of the data source and may modify the source data to reflect this.
        """


class DataTransformer(ABC):

    def __init__(self, config: DataTransformConfig):
        self.config: DataTransformConfig = config

    @abstractmethod
    def apply_transformation(self, raw_data: DataFrame) -> DataFrame:
        """
        This method should transform the raw data and return the resulting dataframe.
        """
        pass


class DataLoader(ABC):

    def __init__(self, config: DataLoadConfig):
        self.config: DataLoadConfig = config

    @abstractmethod
    def load_data(self, data: DataFrame) -> None:
        """
        This method should load the data into the data warehouse.
        """
        pass

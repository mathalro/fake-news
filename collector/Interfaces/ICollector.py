from selenium import webdriver                                                  # allows to initialise a browser
import abc                                                                      # allows abstract class use 

"""
    This interface is a contract to be agreed by any collector
"""
class ICollector(abc.ABC):
    
    @abc.abstractmethod
    def get_news(self):
        pass
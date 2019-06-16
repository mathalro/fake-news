import abc

"""
    This interface is a contract to be agreed by any scraper 
"""
class IScraper(abc.ABC):

    @abc.abstractmethod
    def get_newslist(self, url):
        pass

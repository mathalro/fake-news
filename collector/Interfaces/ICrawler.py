import abc

"""
    This interface is the contract to be agreed by any Crawler
"""
class ICrawler(abc.ABC):

    @abc.abstractmethod
    def get_pagelist(self):
        pass
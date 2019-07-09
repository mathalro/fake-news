from .CrawlerAosfatos import CrawlerAosfatos
from .ScraperAosfatos import ScraperAosfatos
from ..Interfaces.ICollector import ICollector

class CollectorAosfatos(ICollector):

    def __init__(self, url):
        self.url = url

    def get_news(self):
        scraper = CrawlerAosfatos(self.url)
        list_pages = scraper.get_pagelist()

        news_collector = ScraperAosfatos()

        data = []
        for page in list_pages:
            data.extend(news_collector.get_newslist(page))

        return data

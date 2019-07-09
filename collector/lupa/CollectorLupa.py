from .CrawlerLupa import CrawlerLupa
from .ScraperLupa import ScraperLupa
from ..Interfaces.ICollector import ICollector

class CollectorLupa(ICollector):

    def __init__(self, url):
        self.url = url

    def get_news(self):
        scraper = CrawlerLupa(self.url)
        list_pages = scraper.get_pagelist()

        news_collector = ScraperLupa()

        data = []
        for page in list_pages:
            data.extend(news_collector.get_newslist(page))

        return data

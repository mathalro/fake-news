from .CrawlerLupa import CrawlerLupa
from .ScraperLupa import ScraperLupa
from ..Interfaces.ICollector import ICollector

class CollectorLupa(ICollector):

    def get_news(self):
        urlBase = 'https://piaui.folha.uol.com.br/lupa/category/eleicoes-2018/'
        scraper = CrawlerLupa(urlBase)
        list_pages = scraper.get_pagelist()

        for i in list_pages:
            print(i)

        # list_pages = ['https://piaui.folha.uol.com.br/lupa/2018/10/25/debate-witzel-paes-globo/']

        news_collector = ScraperLupa()

        for page in list_pages:
            print(page)
            data = news_collector.get_newslist(page)

        return data

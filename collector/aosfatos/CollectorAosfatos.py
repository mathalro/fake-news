from .CrawlerAosfatos import CrawlerAosfatos
from .ScraperAosfatos import ScraperAosfatos
from ..Interfaces.ICollector import ICollector

class CollectorAosfatos(ICollector):

    def get_news(self):
        urlBase = 'https://aosfatos.org/noticias/eleicoes-2018/'
        scraper = CrawlerAosfatos(urlBase)
        list_pages = scraper.get_pagelist()

        for i in list_pages:
            print(i)

        # list_pages = ['https://piaui.folha.uol.com.br/lupa/2018/10/25/debate-witzel-paes-globo/']

        news_collector = ScraperAosfatos()

        for page in list_pages:
            print(page)
            data = news_collector.get_newslist(page)

        return data

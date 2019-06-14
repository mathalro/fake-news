from .PagesCollectorAosfatos import PagesCollectorAosfatos
from .NewsCollectorAosfatos import NewsCollectorAosfatos

urlBase = 'https://aosfatos.org/noticias/eleicoes-2018/'
scraper = PagesCollectorAosfatos(urlBase)
list_pages = scraper.get_links()

for i in list_pages:
    print(i)

# list_pages = ['https://piaui.folha.uol.com.br/lupa/2018/10/25/debate-witzel-paes-globo/']

news_collector = NewsCollectorAosfatos()

for page in list_pages:
    print(page)
    data = news_collector.get_news(page)

    for sample in data:
        print(sample.content.text)
        print(sample.description.text)
        print(sample.label)
        print()
        

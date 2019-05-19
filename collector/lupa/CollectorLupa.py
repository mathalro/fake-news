from .PagesCollectorLupa import PagesCollectorLupa
from .NewsCollectorLupa import NewsCollectorLupa

urlBase = 'https://piaui.folha.uol.com.br/lupa/category/eleicoes-2018/'
scraper = PagesCollectorLupa(urlBase)
list_pages = scraper.get_links()

#list_pages = ['https://piaui.folha.uol.com.br/lupa/2018/10/25/debate-doria-franca-globo/', 'https://piaui.folha.uol.com.br/lupa/2018/10/27/tudo-sobre-haddad/']

news_collector = NewsCollectorLupa()

for page in list_pages:
    print(page)
    data = news_collector.get_news(page)

    for sample in data:
        print(sample.content.text)
        print(sample.description.text)
        print(sample.label)
        print()
        

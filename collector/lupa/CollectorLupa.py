from .PagesCollectorLupa import PagesCollectorLupa
from .NewsCollectorLupa import NewsCollectorLupa

urlBase = 'https://piaui.folha.uol.com.br/lupa/category/eleicoes-2018/'
scraper = PagesCollectorLupa(urlBase)
list_pages = scraper.get_links()

#list_pages = ['https://piaui.folha.uol.com.br/lupa/2018/10/24/rj-seis-verdades-paes-witzel/']

news_collector = NewsCollectorLupa()

file = open("data", "a+")

for page in list_pages:
    # print(page)
    data = news_collector.get_news(page)

    for sample in data:
        file.write(sample.content.text+"\t"+sample.description.text+"\t"+sample.label+"\n")
        #print(sample.content.text)
        #print(sample.description.text)
        #print(sample.label)
        #print()

file.close()
        

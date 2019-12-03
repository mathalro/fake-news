from .lupa.CollectorLupa import CollectorLupa
from .aosfatos.CollectorAosfatos import CollectorAosfatos

class Collector:

    collectors = []
    collectors.append(CollectorLupa('https://piaui.folha.uol.com.br/lupa/category/eleicoes-2018/'))
    collectors.append(CollectorAosfatos('https://aosfatos.org/noticias/eleicoes-2018/'))

    def Collector(self):
        pass

    def collect(self):

        data = []
        for c in self.collectors:

            print("\nCollecting " + c.__class__.__name__.replace("Collector", ""))

            with open('.data\\' + c.__class__.__name__, 'w') as f:
                news = c.get_news()
                for n in news:
                    f.write(n.content.text+"\t"+n.label+"\n")

        return data

    
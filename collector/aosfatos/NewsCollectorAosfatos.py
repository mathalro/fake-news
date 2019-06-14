import sys

from ..models.newsModels.Article import Article
from ..models.newsModels.Content import Content
from ..models.newsModels.Description import Description

from ..BaseCollector import BaseCollector

"""
	This class collects the links of analytics pages
"""
class NewsCollectorAosfatos(BaseCollector):

	def __init__(self):
		pass

	def get_news(self, url):
		try:
			self.BROWSER.get(url)

			content_list = self.BROWSER.find_elements_by_xpath("//blockquote/p")
			description_list = []
			tags_list = self.BROWSER.find_elements_by_xpath("//blockquote/preceding-sibling::figure/figcaption")

			for i, c in enumerate(content_list):
				split = c.text.split('—')
				if (len(split) < 2):
					split = c.text.split('-')
				content_list[i] = split[0]
				if (len(split) > 1):
					description_list.append(split[1])
				else:
					description_list.append("")

			if (len(content_list) != len(description_list) or len(content_list) != len(tags_list) or len(tags_list) != len(description_list)):
				print("The link " + url + " has inconsistent format")
				return []

			article_list = []

			for i in range(len(content_list)):
				content = Content(text=content_list[i])
				description = Description(text=description_list[i])
				article_list.append(Article(content=content, description=description, label=tags_list[i].text))

			return article_list

		except Exception as e:
			print("ERROR: [NewsCollectorLupa] The collector failed to get news for link: " + url)
			print("Exception: " + str(e))
			return []
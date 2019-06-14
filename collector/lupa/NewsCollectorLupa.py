import sys

from ..models.newsModels.Article import Article
from ..models.newsModels.Content import Content
from ..models.newsModels.Description import Description

from ..BaseCollector import BaseCollector

"""
	This class collects the links of analytics pages
"""
class NewsCollectorLupa(BaseCollector):

	def __init__(self):
		pass

	def get_news(self, url):
		try:
			self.BROWSER.get(url)

			content_list = self.BROWSER.find_elements_by_xpath("//div[contains(@class, 'post-inner')]/div[contains(@class, 'etiqueta')]/preceding-sibling::p/b[contains(text(), '”')]")
			description_list = self.BROWSER.find_elements_by_xpath("//div[contains(@class, 'post-inner')]/div[contains(@class, 'etiqueta')]/preceding-sibling::p/b[contains(text(), '”')]/following-sibling::i[1]")
			tags_list = []

			for c in content_list:
				tags_list.append(c.find_elements_by_xpath("./../following-sibling::div[contains(@class, 'etiqueta')]")[0])

			if (len(content_list) != len(description_list) or len(content_list) != len(tags_list) or len(tags_list) != len(description_list)):
				print("The link " + url + " has inconsistent format")
				return []

			article_list = []

			for i in range(len(content_list)):
				content = Content(text=content_list[i].text)
				description = Description(text=description_list[i].text)
				article_list.append(Article(content=content, description=description, label=tags_list[i].text))

			return article_list

		except Exception as e:
			print("ERROR: [NewsCollectorLupa] The collector failed to get news for link: " + url)
			print("Exception: " + e)
			return []
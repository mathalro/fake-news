import sys

from ..BaseCollector import BaseCollector
from ..Interfaces.ICrawler import ICrawler

"""
	This class extracts all the news of analysis page and separate the attributes of each news
"""
class CrawlerLupa(ICrawler, BaseCollector):

	def __init__ (self, url):
		self.url = url

	def get_pagelist(self):
		
		print("Initialize browser")

		try:
			print("Initialize")

			button = ""
			link = self.url
			news_links = []
			find = True
			
			print("Get links")
			
			while find == True:
				try:
					self.BROWSER.get(link)
					elements = self.BROWSER.find_elements_by_xpath('.//div[@class="bloco"]//a[@class="bloco-img"]')
					
					for a in elements:
						news_links.append(a.get_attribute('href'))
						
					button = self.BROWSER.find_elements_by_xpath('.//a[@class="btn-mais btnvermais"]')
					if (len(button) == 0):
						find = False
					else:
						link = button[0].get_attribute('href')
						
				except Exception as ex:
					print(ex)
					sys.exit(1)

			print("Got links")

			return news_links
		except Exception as ex:
			print(ex)
			sys.exit(1)
from selenium import webdriver                                                  # allows to initialise a browser
from selenium.webdriver.common.by import By                                     # allows to search for things
from selenium.webdriver.support.ui import WebDriverWait                         # allows to wait for a page load
from selenium.webdriver.support import expected_conditions as EC                # specify what looking for
from selenium.common.exceptions import TimeoutException, NoSuchElementException # handling timeout situation

"""
    Base Web Scraper
"""
class BaseCollector:
    BROWSER = webdriver.Chrome('./dependencies/chromedriver.exe')

# Fake News Identification for Brazilian Politics [![CircleCI](https://circleci.com/gh/matheus-almeida-rosa/fake-news.svg?style=svg&circle-token=bfd9c86a73d41dd7af6df237e5bc8804869816f4)](https://circleci.com/gh/matheus-almeida-rosa/fake-news)

## About the project

This project is part of a **Final Paper** for the course of **Computer Engineering** in **Federal Center of Tecnological Education of Minas Gerais**. 

## Objective

The project aims to identify possible lies inside Brazilian politics text or phases, in order to try to face the fake news problem.

## To contribute

Due to great need of large amount of data, the main contribution to be made for this project is on collecting labeled news data. In order to standardize the contributions, it was defined a pattern for Collector creation:

* An collector should be created for reliable news sites, or depending of the site size, for the site sections;
* An collector should be created, if and only if, there is no collector for that site/page;
* Every collector should use some kind of Web crawler to index the web pages;
* Every collector should use some kind of Web Scraper to extract information from the indexed pages;
* Every collector should return data in the Article format.

The above rules are represented by the following class diagram:

![alt text](https://drive.google.com/uc?export=view&id=1n5dLiSC0mbHsLHLL3v7CErTBi_EV2kHk)

# Configuration

## Requirements

```
pip3 install -r requirements
```

## Dependencies

chromedrive: http://chromedriver.chromium.org/downloads

# Run

```
python3 main.py
```

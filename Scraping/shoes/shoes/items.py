# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ShoesItem(scrapy.Item):
    # Define the fields for your item here like:
    kind = scrapy.Field()
    brand = scrapy.Field()
    model = scrapy.Field()
    file_urls = scrapy.Field()

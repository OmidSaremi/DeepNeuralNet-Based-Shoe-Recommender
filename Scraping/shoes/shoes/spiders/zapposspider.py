from shoes.items import ShoesItem
import datetime
import scrapy

starting_url = "http://www.zappos.com/mens-shoes?pf_rd_r=0625X0PC4JBF736ZBYJ0&pf_rd_p=c6b62617-0229-4d70-9725-06e887895ddf"
X = ["http://www.zappos.com/men-oxfords/CK_XARC31wHAAQLiAgMYAQI.zso?s=goliveRecentSalesStyle/desc/&pf_rd_r=0AZR7F1Z73YSNT2JXRVJ&pf_rd_p=7ab7e484-b6f3-4147-a37a-0149a357dd47", "http://www.zappos.com/mens-sandals~37?s=goliveRecentSalesStyle/desc/&pf_rd_r=0AZR7F1Z73YSNT2JXRVJ&pf_rd_p=7ab7e484-b6f3-4147-a37a-0149a357dd47"]

class ZapposSpider(scrapy.Spider):
    name = "zapposspider"
    # Shoes page
    start_urls = [starting_url]

    def parse(self, response):
        # Men's boot page
        for url in X:
            yield scrapy.Request(url, self.parse_page)

    def parse_page(self, response):
		# Loop over all shoe link elements that link off to the wanted image
		# and yield a request to grab the shoe
		# data and image
        for href in response.xpath("//*[@id='searchResults']/a"):
            yield scrapy.Request(''.join(['http://www.zappos.com', href.xpath("@href").extract_first()]), self.parse_shoes)

		# Extract the 'Next' link from the pagination, load it, and
		# parse it
        next = response.xpath("//a[contains(@class, 'btn secondary')]//@href")
        for url in next.extract():
            yield scrapy.Request(''.join('http://www.zappos.com', url, self.parse_page))

    def parse_shoes(self, response):
        # Grab the URL of the shoe image
        img = response.xpath("//*[@id='angle-3']").xpath("@href")
        imageURL = ''.join(['http://www.zappos.com', img.extract_first()])
        kind = response.xpath('//*[@id="breadcrumbs"]/a[2]/text()').extract_first()
        brand = response.xpath('//*[@id="prdImage"]/h1/a[1]/text()').extract_first()
        model = response.xpath('//*[@id="prdImage"]/h1/a[2]/span/text()').extract_first()
		# yield the result
        yield ShoesItem(kind=kind, brand=brand, model=model, file_urls=[imageURL])

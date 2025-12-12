#!/usr/bin/env python3
"""
Kavita Kosh Crawler - Extract poetry data from https://kavitakosh.org
Designed for NLP research on North Indian languages (Hindi, Urdu, Marathi, Bengali, etc.)
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import time
from datetime import datetime


class KavitakoshSpider(CrawlSpider):
    """
    Spider to crawl Kavita Kosh poetry repository
    Extracts poems with metadata (author, language, category, etc.)
    """
    
    name = 'kavitakosh'
    allowed_domains = ['kavitakosh.org']
    start_urls = [
        'https://kavitakosh.org/kk/index.php?title=%E0%A4%A8%E0%A4%BF%E0%A4%B0%E0%A5%8D%E0%A4%AD%E0%A4%A4%E0%A4%B0_%E0%A4%95%E0%A4%BE%E0%A4%B5%E0%A4%AF'  # Main page
    ]
    
    # Rules for following links
    rules = (
        # Follow links to poet pages
        Rule(
            LinkExtractor(
                allow=r'/kk/index\.php\?title=',
                deny=[r'diff=', r'oldid=', r'action=']  # Skip edit/diff pages
            ),
            callback='parse_page',
            follow=True
        ),
    )
    
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (compatible; NLP-Research-Bot/1.0)',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 1,  # Respectful crawling
        'DOWNLOAD_DELAY': 2,  # 2-second delay between requests
        'RETRY_TIMES': 2,
        'COOKIES_ENABLED': False,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = Path('kavitakosh_data')
        self.output_dir.mkdir(exist_ok=True)
        self.metadata_file = self.output_dir / 'metadata.jsonl'
        self.poems_counter = 0
        
    def parse_page(self, response):
        """Parse individual poem/author/category pages"""
        
        page_title = response.css('h1.firstHeading::text').get()
        
        if not page_title:
            return
        
        # Extract main content
        content_div = response.css('div#mw-content-text')
        
        if not content_div:
            return
        
        # Get poem text (usually in <p> or <div> tags)
        poem_text = self._extract_poem_text(content_div)
        
        if not poem_text or len(poem_text.strip()) < 20:
            return  # Skip if too short
        
        # Extract metadata
        metadata = self._extract_metadata(response, page_title, poem_text)
        
        if metadata:
            self._save_poem(page_title, poem_text, metadata)
            self.poems_counter += 1
            
            if self.poems_counter % 10 == 0:
                self.logger.info(f"Downloaded {self.poems_counter} poems")
    
    def _extract_poem_text(self, content_div):
        """Extract clean poem text from content div"""
        # Remove script and style elements
        for elem in content_div.xpath('.//script | .//style'):
            elem.extract()
        
        # Get all text content
        paragraphs = content_div.css('p::text, div.poem-text::text, div.poem::text').getall()
        poem_text = '\n'.join([p.strip() for p in paragraphs if p.strip()])
        
        return poem_text or ''
    
    def _extract_metadata(self, response, page_title, poem_text):
        """Extract metadata about the poem/author"""
        metadata = {
            'title': page_title,
            'url': response.url,
            'downloaded_at': datetime.now().isoformat(),
            'language': self._detect_language(page_title, poem_text),
            'content_length': len(poem_text),
        }
        
        # Try to extract category/classification
        categories = response.css('a[title^="Category:"]::text').getall()
        if categories:
            metadata['categories'] = categories
        
        # Look for infobox data (author info, etc.)
        infobox = response.css('table.infobox')
        if infobox:
            metadata['has_infobox'] = True
            # Extract key-value pairs from infobox
            rows = infobox.css('tr')
            for row in rows:
                header = row.css('th::text').get()
                value = row.css('td::text').get()
                if header and value:
                    metadata[header.strip()] = value.strip()
        
        return metadata
    
    def _detect_language(self, title, text):
        """Detect language based on script"""
        devanagari_range = range(0x0900, 0x097F)
        
        # Check if text contains Devanagari script
        has_devanagari = any(ord(c) in devanagari_range for c in text)
        has_devanagari_title = any(ord(c) in devanagari_range for c in title)
        
        if has_devanagari or has_devanagari_title:
            # Could be Hindi, Marathi, Sanskrit, etc.
            # Heuristic: check for common Marathi/Hindi suffixes
            if any(marathi_indicator in text for marathi_indicator in ['ी', 'या', 'ें']):
                return 'Devanagari (Hindi/Marathi/Sanskrit)'
            return 'Devanagari'
        
        # Check for Urdu/Persian script
        urdu_range = range(0x0600, 0x06FF)
        if any(ord(c) in urdu_range for c in text):
            return 'Urdu'
        
        return 'Unknown'
    
    def _save_poem(self, title, text, metadata):
        """Save poem and its metadata"""
        # Create filename from title
        safe_filename = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length
        
        # Save poem text
        poem_file = self.output_dir / f"{safe_filename}_{self.poems_counter}.txt"
        with open(poem_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        metadata['text_file'] = str(poem_file)
        
        # Append metadata to JSONL file
        with open(self.metadata_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')


class KavitakoshDirectSpider(scrapy.Spider):
    """
    Alternative spider using direct URLs for faster targeted crawling
    Use this if you know specific author/poem pages
    """
    
    name = 'kavitakosh_direct'
    allowed_domains = ['kavitakosh.org']
    
    def __init__(self, authors_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.authors_file = authors_file
        self.output_dir = Path('kavitakosh_data')
        self.output_dir.mkdir(exist_ok=True)
    
    def start_requests(self):
        """Generate requests for specific pages"""
        # Example: known poets in North Indian languages
        poets = [
            'कबीर',  # Kabir
            'तुलसीदास',  # Tulsidas
            'सूरदास',  # Surdas
            'मीराबाई',  # Meerabai
            'रहीम',  # Rahim
        ]
        
        for poet in poets:
            from urllib.parse import quote
            encoded_poet = quote(poet.encode('utf-8'))
            url = f'https://kavitakosh.org/kk/index.php?title={encoded_poet}'
            yield scrapy.Request(url, callback=self.parse_poet)
    
    def parse_poet(self, response):
        """Parse poet's works"""
        poet_name = response.css('h1.firstHeading::text').get()
        
        # Extract all poems/works linked from this page
        poem_links = response.css('a[title*="कविता"], a[title*="poem"]')
        
        for link in poem_links:
            poem_url = response.urljoin(link.attrib.get('href'))
            yield scrapy.Request(poem_url, callback=self.parse_poem_page)
    
    def parse_poem_page(self, response):
        """Parse individual poem"""
        # Similar to KavitakoshSpider.parse_page
        poem_title = response.css('h1.firstHeading::text').get()
        poem_text = '\n'.join(
            response.css('div#mw-content-text p::text').getall()
        )
        
        if poem_text and poem_title:
            safe_filename = "".join(c for c in poem_title if c.isalnum() or c in (' ', '_')).rstrip()
            output_file = self.output_dir / f"{safe_filename}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {poem_title}\n\n")
                f.write(poem_text)


def run_crawler(spider_class='KavitakoshSpider', concurrent_requests=1):
    """Run the Scrapy crawler"""
    
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (compatible; NLP-Research-Bot/1.0)',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': concurrent_requests,
        'DOWNLOAD_DELAY': 2,
        'RETRY_TIMES': 2,
        'LOG_LEVEL': 'INFO',
        'COOKIES_ENABLED': False,
        'TELNETCONSOLE_ENABLED': False,
    })
    
    if spider_class == 'KavitakoshSpider':
        process.crawl(KavitakoshSpider)
    else:
        process.crawl(KavitakoshDirectSpider)
    
    print(f"Starting crawler: {spider_class}")
    print(f"Output directory: {Path('kavitakosh_data').absolute()}")
    process.start()


if __name__ == '__main__':
    # Run the crawler
    run_crawler('KavitakoshDirectSpider')

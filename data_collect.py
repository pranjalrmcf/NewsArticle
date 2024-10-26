import newspaper
import pandas as pd

# List of Indian newspaper URLs and corresponding newspaper names
newspaper_data = {
    'Hindustan Times': 'https://www.hindustantimes.com',
    'Indian Express': 'https://indianexpress.com',
    'Telegraph India': 'https://www.telegraphindia.com',
    'Deccan Chronicle': 'https://www.deccanchronicle.com',
    'New Indian Express': 'https://www.newindianexpress.com',
    'Live Mint': 'https://www.livemint.com',
    'Business Standard': 'https://www.business-standard.com',
    'Financial Express': 'https://www.financialexpress.com',
    'DNA India': 'https://www.dnaindia.com',
    'The Tribune': 'https://www.tribuneindia.com',
    'The Statesman': 'https://www.thestatesman.com',
    'Asian Age': 'https://www.asianage.com',
    'Daily Pioneer': 'https://www.dailypioneer.com',
    'Free Press Journal': 'https://www.freepressjournal.in',
    'Economic Times': 'https://economictimes.indiatimes.com',
    'The Hans India': 'https://www.thehansindia.com',
    'Orissa Post': 'https://www.orissapost.com',
    'The Hitavada': 'https://www.thehitavada.com',
    'The Sentinel Assam': 'https://www.sentinelassam.com',
    'Navhind Times': 'https://www.navhindtimes.in',
    'Assam Tribune': 'https://assamtribune.com',
    'Arunachal Times': 'https://arunachaltimes.in',
    'Shillong Times': 'https://theshillongtimes.com',
    'Sanga Express': 'https://www.thesangaiexpress.com'
}

# Fetch articles only for the selected newspaper
def fetch_news_articles(selected_newspaper, article_limit):
    articles_list = []
    url = newspaper_data.get(selected_newspaper)

    if not url:
        print(f"No URL found for {selected_newspaper}")
        return pd.DataFrame()

    try:
        print(f"Building newspaper {selected_newspaper} from URL: {url}")
        paper = newspaper.build(url, language='en', memoize_articles=False)

        # Check if any articles are found
        if len(paper.articles) == 0:
            print(f"No articles found for {selected_newspaper}.")
            return pd.DataFrame()

        print(f"Found {len(paper.articles)} articles in {selected_newspaper}")
        article_count = 0

        for article in paper.articles:
            if article_count >= article_limit:
                break

            try:
                article.download()
                article.parse()
                article.nlp()

                # Prepare published date
                published_date = article.publish_date.strftime('%Y-%m-%d') if article.publish_date else None

                # Append to the articles list
                articles_list.append({
                    'Newspaper Name': selected_newspaper,
                    'Published Date': published_date,
                    'URL': article.url,
                    'Headline': article.title,
                    'Content': article.text,
                })
                article_count += 1
            except Exception as e:
                print(f"Error processing article {article.url}: {e}")
                continue

    except Exception as e:
        print(f"Error fetching articles from {selected_newspaper}: {e}")

    return pd.DataFrame(articles_list)

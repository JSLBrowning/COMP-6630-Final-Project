from urllib.request import urlopen

url = "https://www.songlyrics.com/needtobreathe/testify-lyrics/"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
print(html)

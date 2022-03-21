import codecs
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


def _parse_to_soup(url):
    print(url)
    html_req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html_raw = urlopen(html_req).read().decode('utf-8')
    html_soup = BeautifulSoup(html_raw, 'html.parser')
    return html_soup


def _parse_genres(url):
    home = _parse_to_soup(url)
    parsed_genres = []
    for a in home.find_all('a'):
        if str(a.get('href'))[0].isalpha() and (str(a.get('href')).endswith('-lyrics.php')) and not (str(a.get('href')).startswith('latin')):
            parsed_genres.append([a.get('href').replace('-lyrics.php', '').replace('-', '_'), a.get('href')])
    print(parsed_genres)
    return parsed_genres


def _clean_list(song_list):
    return [list(i) for i in {*[tuple(sorted(i)) for i in song_list]}]


def _scrape_lyrics(root_url, genres_to_scrape):
    for g in genres_to_scrape:
        songs = []
        genre_soup = _parse_to_soup(root_url + '/' + g[1])
        for a in genre_soup.find_all('a'):
            if str(a.get('href')).endswith('-lyrics/') and str(a.get('href')).startswith('http'):
                song = []
                song_soup = _parse_to_soup(a.get('href'))
                try:
                    pagetitle = song_soup.find('div', {'class': 'pagetitle'})
                    pagetitle_h1 = pagetitle.findChild('h1', recursive=False)
                    artist_and_title = str(pagetitle_h1.get_text()).split(' - ')
                    song.append('TITLE: ' + artist_and_title[1])
                    song.append('ARTIST: ' + artist_and_title[0])
                    song.append('LYRICS:\n' + str(song_soup.find('p', {'id': 'songLyricsDiv'}).get_text()))
                    songs.append(song)
                except Exception:
                    print('Skipping.')
        songs = _clean_list(songs)
        f = codecs.open('baseline/genre_' + g[0] + '.txt', 'w+', 'utf-8')
        for s in songs:
            for d in s:
                f.write(d + '\n')
        f.write('/END LYRICS\n')
        f.close()


genres = _parse_genres('https://www.songlyrics.com/musicgenres.php')
_scrape_lyrics('https://www.songlyrics.com', genres)

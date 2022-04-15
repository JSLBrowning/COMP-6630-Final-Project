import numpy as np
import re


class DataLoader:

    # lowResAmount is the percentage that we
    # would like to reduce the
    def __init__(self, low_res_amount=.25):
        self.data = []
        self.allSongs = []
        self.songWordDict = {}  # a dictionary of unique lyric words
        self.songWordDictScaled = {}
        self.uniqueWordCount = None
        self.numpy_arrays = []
        self.maxDimension = None
        self.lowResAmount = low_res_amount

    def loadData(self):
        singleSong = []

        textLines = open('../data/baseline/genre_country_music.txt').readlines()

        i = 1
        for line in textLines:
            words = line.split(' ')
            if (words[0].strip() == 'TITLE:' or words[0].strip() == 'ARTIST:' or
                    words[0].strip() == 'LYRICS:' or words[0].strip() == '/END' or
                    words[0].strip() == '' or words[0].strip()[0] == '['):
                if words[0].strip() == '/END':
                    self.allSongs.append(singleSong)
                    singleSong = []
                continue
            else:
                for word in words:
                    w = word.lower().strip()
                    w = re.sub(r'[^\w\s]', '', w)
                    singleSong.append(w)
                    if not w in self.songWordDict.keys():
                        self.songWordDict.update({w: i})
                        i += 1

        self.uniqueWordCount = i - 1
        self.createNumArrs()
        self.createScaledWordDict()

    def getData(self):
        return self.data

    def getUniqueWordCount(self):
        return self.uniqueWordCount

    def getMaxDimension(self):
        return self.maxDimension

    def getSongWordDict(self, scaled=False):
        if not scaled:
            return self.songWordDict
        else:
            return self.songWordDictScaled

    def findMaxSongLength(self):
        maxSongLength = 0
        for song in self.allSongs:
            if len(song) > maxSongLength:
                maxSongLength = len(song)

        #print(maxSongLength)
        return maxSongLength

    def createScaledWordDict(self):
        for k, v in self.songWordDict.items():
            self.songWordDictScaled[k] = v / self.uniqueWordCount

    def createNumArrs(self):

        maxSongLength = self.findMaxSongLength()
        self.maxDimension = self.getDimensions(maxSongLength)
        maxArrLen = pow(self.maxDimension, 2)
        # print('Max np array dimension: ', maxDimension, 'x', maxDimension, sep='')

        for i in range(len(self.allSongs)):
            songLength = len(self.allSongs[i])
            songNArr = np.zeros(maxArrLen, dtype=int)
            for h in range(songLength):
                word = self.allSongs[i][h]
                # get the unique identifier from the dict
                wordNum = self.songWordDict[word]
                # set this index in the numpy array to be that number
                songNArr[h] = wordNum
            self.data.append(songNArr)

        self.data = np.array(self.data)

        self.data = self.data.reshape(self.data.shape[0], self.maxDimension, self.maxDimension)

    def getDimensions(self, a, i=0):
        if pow(i, 2) > a:
            while not (i % (1 / self.lowResAmount) == 0):
                i += 1
            #print(i)
            return i
        else:
            return self.getDimensions(a, i + 1)


if __name__ == '__main__':
    dl = DataLoader()

    dl.loadData()

    print(dl.data.shape)

    # print(dl.data[0])

    # print(dl.songWordDict)
    print(dl.getUniqueWordCount())
# print(dl.getSongWordDict(scaled=True))

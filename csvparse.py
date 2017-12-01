import csv

'''
Data is split between channels on first dimension (data[0] is speaker 1, data [1] is speaker 2).
Each data item is [STARTTIME, SPEAKER, SPEAKING/NOT (S/N), ENDTIME]

USAGE SAMPLE:

from csvparse import loadCSV

if __name__ == '__main__':
    print(loadCSV('/data/CSV_Files_from_Praat/HS_D01.csv'))
'''

def loadCSV(path):
    with open("data/CSV_Files_from_Praat/HS_D01.csv") as csvfile:
        data = list(csv.reader(csvfile))[1:-1]
        channels = [[x for x in data if x[1] == data[0][1]], [y for y in data if y[1] == data[1][1]]]
    return channels
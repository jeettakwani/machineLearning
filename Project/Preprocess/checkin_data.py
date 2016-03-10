__author__ = 'jtakwani'

from types import *

import json
import csv
import pandas as pd
from datetime import datetime

def convertToCsv(input_file, output_file,columnsNeeded,idColumn):


    json_data = []
    data = None
    write_header = True
    item_keys = []
    data = []
    count = 0
    counter = 0
    with open(input_file) as json_file:
        for line in json_file:
            count += 1
            data.append(json.loads(line))

    with open(output_file, 'wb') as csv_file:
        writer = csv.writer(csv_file)

        for item in data:
            item_values = []
            if item['business_id'] not in idColumn:
                counter +=1
                continue
            for key in item:
                if write_header:

                    if key not in columnsNeeded:
                        continue

                    if key == 'checkin_info':
                        item_keys.append('sunday')
                        item_keys.append('monday')
                        item_keys.append('tuesday')
                        item_keys.append('wednesday')
                        item_keys.append('thursday')
                        item_keys.append('friday')
                        item_keys.append('saturday')
                    else:
                        item_keys.append(key)

                if key not in columnsNeeded:
                    continue

                if key == 'checkin_info':

                    value = processInfo(item[key])

                    for i in range(len(value)):
                        item_values.append(value[i])

                else:
                    value = item.get(key, '')
                    if isinstance(value, StringTypes):
                        item_values.append(value.encode('utf-8'))
                    else:
                        item_values.append(value)

            if write_header:
                writer.writerow(item_keys)
                write_header = False

            writer.writerow(item_values)
        print counter

def processInfo(items):
    count = [0]*7

    for item in items:
        key = item.split('-')
        count[int(key[1])] += items[item]

    return count



def main():

    input_file = '../dataset/checkin.json'
    train_file = '../csv/train_checkin.csv'
    test_file = '../csv/test_checkin.csv'

    columnsNeeded = ['business_id','checkin_info']

    train_business_data = pd.read_csv('../csv/train_business.csv')
    test_business_data = pd.read_csv('../csv/test_business.csv')

    trainIdColumn = list(train_business_data['business_id'])
    testIdColumn = list(test_business_data['business_id'])

    convertToCsv(input_file,train_file,columnsNeeded,trainIdColumn)
    convertToCsv(input_file,test_file,columnsNeeded,testIdColumn)


if __name__ == "__main__":
    main()
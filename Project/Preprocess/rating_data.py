__author__ = 'jtakwani'

__author__ = 'jtakwani'

from types import *

import json
import csv
import pandas as pd
from datetime import datetime

def convertToCsv(input_file, output_file, columnsNeeded,idColumn):


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
            flag = 0
            if item['business_id'] not in idColumn:
                counter +=1
                #print item['business_id']
                continue
            for key in item:
                if write_header:
                    if key not in columnsNeeded:
                        continue
                    item_keys.append(key)
                    if key == 'date':
                        item_keys.append('year')
                        item_keys.append('month')
                        item_keys.append('day')
                        item_keys.append('dayOfWeek')

                if key not in columnsNeeded:
                    continue


                value = item.get(key, '')

                if key == 'date':

                    if isinstance(value, StringTypes):
                        item_values.append(value.encode('utf-8'))
                    else:
                        item_values.append(value)

                    dt = datetime.strptime(item[key], '%Y-%m-%d')
                    item_values.append(dt.year)
                    item_values.append(dt.month)
                    item_values.append(dt.day)
                    dayOfWeek = datetime(dt.year,dt.month,dt.day).isoweekday()
                    if dayOfWeek == 7:
                        dayOfWeek = 0
                    item_values.append(dayOfWeek)

                else:
                    if isinstance(value, StringTypes):
                        item_values.append(value.encode('utf-8'))
                    else:
                        item_values.append(value)

            if write_header:
                writer.writerow(item_keys)
                write_header = False

            writer.writerow(item_values)
        print counter


def main():

    input_file = '../dataset/review.json'
    output_file = '../csv/review.csv'

    columnsNeeded = ['business_id','date','stars']

    business_data = pd.read_csv('../csv/business.csv')

    idColumn = list(business_data['business_id'])

    convertToCsv(input_file,output_file,columnsNeeded,idColumn)

if __name__ == "__main__":
    main()
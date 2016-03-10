__author__ = 'jtakwani'

from types import *

import json
import csv

def convertToCsv(input_file, output_file, columnsNeeded,f):

    json_data = []
    data = None
    write_header = True
    item_keys = []
    data = []
    count = 0
    counter = 0
    with open(input_file) as json_file:
        for line in json_file:
            data.append(json.loads(line))

    with open(output_file, 'wb') as csv_file:
        writer = csv.writer(csv_file)

        for item in data:
            item_values = []
            flag = 0
            counter +=1
            for key in item:
                if write_header:
                    if key not in columnsNeeded:
                        continue
                    item_keys.append(key)

                if key not in columnsNeeded:
                    continue

                if key == 'categories':

                    if 'nightlife' in str(item[key]).lower()\
                            or 'bar' in str(item[key]).lower()\
                            or 'bars' in str(item[key]).lower()\
                            or 'pub' in str(item[key]).lower():

                        value = 'nightlife'

                    elif 'restaurant' in item[key] \
                            or 'Restaurants' in item[key]:

                        value = 'restaurant'

                    else:
                        flag = 1
                        continue

                else:
                    value = item.get(key, '')

                if isinstance(value, StringTypes):
                    item_values.append(value.encode('utf-8'))
                else:
                    item_values.append(value)

            if write_header:
                writer.writerow(item_keys)
                write_header = False

            if not flag and f =='train' and count != 5000:
                writer.writerow(item_values)
                count +=1

            elif not flag and f == 'test' and count < 5000 :
                count +=1
                print counter
            elif not flag and f == 'test' and count != 7001:
                writer.writerow(item_values)
                count +=1


            else:

                continue


def main():

    input_file = '../dataset/business.json'
    train_file = '../csv/train_business.csv'
    test_file = '../csv/test_business.csv'

    columnsNeeded = ['attributes','business_id','categories','name','open','stars']

    convertToCsv(input_file,train_file,columnsNeeded,'train')
    convertToCsv(input_file,test_file,columnsNeeded,'test')
if __name__ == "__main__":
    main()
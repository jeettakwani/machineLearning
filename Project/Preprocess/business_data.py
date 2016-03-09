__author__ = 'jtakwani'

from types import *

import json
import csv

def convertToCsv(input_file, output_file, columnsNeeded):


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
            counter += 1
            item_values = []
            flag = 0
            for key in item:
                if write_header:
                    if key not in columnsNeeded:
                        continue
                    item_keys.append(key)

                if key not in columnsNeeded:
                    continue

                if key == 'categories':

                    if 'restaurant' in item[key] \
                            or 'Restaurants' in item[key]:

                        value = 'restaurant'

                    elif 'nightlife' in str(item[key]).lower():

                        value = 'nightlife'

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

            if not flag:
                writer.writerow(item_values)
                count +=1
                #print count
                if count == 5001:
                    break



def main():

    input_file = '../dataset/business.json'
    output_file = '../csv/business.csv'

    columnsNeeded = ['attributes','business_id','categories','name','open','stars']

    convertToCsv(input_file,output_file,columnsNeeded)
if __name__ == "__main__":
    main()
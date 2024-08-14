import re

def extract_number(s):
    number = re.findall('\d+', s)
    return int(number[0]) if number else 0
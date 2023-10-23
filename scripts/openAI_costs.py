#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:10:03 2023

@author: dgaio
"""

import re

text = """
2:00 PM
Local time: Oct 20, 2023, 4:00 PM
gpt-3.5-turbo-16k-0613, 13 requests
20,387 prompt + 485 completion = 20,872 tokens
2:10 PM
Local time: Oct 20, 2023, 4:10 PM
gpt-3.5-turbo-16k-0613, 42 requests
69,300 prompt + 1,769 completion = 71,069 tokens
2:15 PM
Local time: Oct 20, 2023, 4:15 PM
gpt-3.5-turbo-16k-0613, 40 requests
64,601 prompt + 1,522 completion = 66,123 tokens
2:25 PM
Local time: Oct 20, 2023, 4:25 PM
gpt-3.5-turbo-16k-0613, 42 requests
72,539 prompt + 2,094 completion = 74,633 tokens
2:30 PM
Local time: Oct 20, 2023, 4:30 PM
gpt-3.5-turbo-16k-0613, 44 requests
71,858 prompt + 1,788 completion = 73,646 tokens
2:35 PM
Local time: Oct 20, 2023, 4:35 PM
gpt-3.5-turbo-16k-0613, 55 requests
122,529 prompt + 3,165 completion = 125,694 tokens
2:40 PM
Local time: Oct 20, 2023, 4:40 PM
gpt-3.5-turbo-16k-0613, 51 requests
117,889 prompt + 2,826 completion = 120,715 tokens
2:45 PM
Local time: Oct 20, 2023, 4:45 PM
gpt-3.5-turbo-16k-0613, 64 requests
112,003 prompt + 3,192 completion = 115,195 tokens
2:50 PM
Local time: Oct 20, 2023, 4:50 PM
gpt-3.5-turbo-16k-0613, 116 requests
138,190 prompt + 3,479 completion = 141,669 tokens
2:55 PM
Local time: Oct 20, 2023, 4:55 PM
gpt-3.5-turbo-16k-0613, 8 requests
10,567 prompt + 270 completion = 10,837 tokens
3:00 PM
530 requests
3:05 PM
Local time: Oct 20, 2023, 5:05 PM
gpt-3.5-turbo-16k-0613, 80 requests
116,000 prompt + 2,914 completion = 118,914 tokens
3:10 PM
Local time: Oct 20, 2023, 5:10 PM
gpt-3.5-turbo-16k-0613, 34 requests
113,460 prompt + 2,656 completion = 116,116 tokens
3:15 PM
Local time: Oct 20, 2023, 5:15 PM
gpt-3.5-turbo-16k-0613, 34 requests
117,338 prompt + 2,680 completion = 120,018 tokens
3:20 PM
Local time: Oct 20, 2023, 5:20 PM
gpt-3.5-turbo-16k-0613, 15 requests
47,483 prompt + 1,254 completion = 48,737 tokens
3:30 PM
Local time: Oct 20, 2023, 5:30 PM
gpt-3.5-turbo-16k-0613, 200 requests
154,188 prompt + 4,212 completion = 158,400 tokens
3:35 PM
Local time: Oct 20, 2023, 5:35 PM
gpt-3.5-turbo-16k-0613, 102 requests
87,181 prompt + 3,065 completion = 90,246 tokens
3:40 PM
Local time: Oct 20, 2023, 5:40 PM
gpt-3.5-turbo-16k-0613, 22 requests
110,335 prompt + 2,804 completion = 113,139 tokens
3:45 PM
Local time: Oct 20, 2023, 5:45 PM
gpt-3.5-turbo-16k-0613, 23 requests
117,452 prompt + 2,895 completion = 120,347 tokens
3:50 PM
Local time: Oct 20, 2023, 5:50 PM
gpt-3.5-turbo-16k-0613, 14 requests
72,735 prompt + 2,021 completion = 74,756 tokens
3:55 PM
Local time: Oct 20, 2023, 5:55 PM
gpt-3.5-turbo-16k-0613, 6 requests
42,350 prompt + 1,092 completion = 43,442 tokens
4:00 PM
35 requests
4:00 PM
Local time: Oct 20, 2023, 6:00 PM
gpt-3.5-turbo-16k-0613, 16 requests
115,144 prompt + 2,639 completion = 117,783 tokens
4:05 PM
Local time: Oct 20, 2023, 6:05 PM
gpt-3.5-turbo-16k-0613, 8 requests
65,977 prompt + 1,430 completion = 67,407 tokens
4:15 PM
Local time: Oct 20, 2023, 6:15 PM
gpt-3.5-turbo-16k-0613, 9 requests
65,765 prompt + 1,669 completion = 67,434 tokens
4:20 PM
Local time: Oct 20, 2023, 6:20 PM
gpt-3.5-turbo-16k-0613, 2 requests
12,692 prompt + 310 completion = 13,002 tokens
"""


# Regular expressions to match the number of requests, prompt tokens, and completion tokens
requests_pattern = r"(?<!AM|PM)\s(\d+)\srequests"
prompt_pattern = r"(\d+,\d+)\sprompt"
completion_pattern = r"(\d+,\d+)\scompletion"

# Extract the values
requests_matches = re.findall(requests_pattern, text)
prompt_matches = re.findall(prompt_pattern, text)
completion_matches = re.findall(completion_pattern, text)

# Convert comma-separated strings to integers
def convert_to_int(match):
    return int(match.replace(",", ""))

total_requests = sum([convert_to_int(match) for match in requests_matches])
total_prompt_tokens = sum([convert_to_int(match) for match in prompt_matches])
total_completion_tokens = sum([convert_to_int(match) for match in completion_matches])

print(f"Total requests: {total_requests}")
print(f"Total prompt tokens: {total_prompt_tokens}")
print(f"Total completion tokens: {total_completion_tokens}")




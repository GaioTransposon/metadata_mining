#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:10:03 2023

@author: dgaio
"""

import re

text = """
11:00 AM
Local time: Oct 23, 2023, 1:00 PM
gpt-3.5-turbo-16k-0613, 28 requests
76,788 prompt + 2,246 completion = 79,034 tokens
11:05 AM
Local time: Oct 23, 2023, 1:05 PM
gpt-3.5-turbo-16k-0613, 44 requests
120,626 prompt + 2,753 completion = 123,379 tokens
11:10 AM
Local time: Oct 23, 2023, 1:10 PM
gpt-3.5-turbo-16k-0613, 42 requests
114,664 prompt + 2,904 completion = 117,568 tokens
11:15 AM
Local time: Oct 23, 2023, 1:15 PM
gpt-3.5-turbo-16k-0613, 13 requests
35,935 prompt + 958 completion = 36,893 tokens
11:20 AM
Local time: Oct 23, 2023, 1:20 PM
gpt-3.5-turbo-16k-0613, 46 requests
121,436 prompt + 3,090 completion = 124,526 tokens
11:25 AM
Local time: Oct 23, 2023, 1:25 PM
gpt-3.5-turbo-16k-0613, 2 requests
4,787 prompt + 54 completion = 4,841 tokens
11:35 AM
Local time: Oct 23, 2023, 1:35 PM
gpt-3.5-turbo-16k-0613, 44 requests
125,329 prompt + 3,054 completion = 128,383 tokens
11:40 AM
Local time: Oct 23, 2023, 1:40 PM
gpt-3.5-turbo-16k-0613, 26 requests
71,920 prompt + 1,947 completion = 73,867 tokens
11:45 AM
Local time: Oct 23, 2023, 1:45 PM
gpt-3.5-turbo-16k-0613, 44 requests
114,829 prompt + 2,776 completion = 117,605 tokens
11:50 AM
Local time: Oct 23, 2023, 1:50 PM
gpt-3.5-turbo-16k-0613, 43 requests
123,343 prompt + 2,959 completion = 126,302 tokens
11:55 AM
Local time: Oct 23, 2023, 1:55 PM
gpt-3.5-turbo-16k-0613, 43 requests
117,712 prompt + 2,838 completion = 120,550 tokens
12:00 PM
434 requests
12:00 PM
Local time: Oct 23, 2023, 2:00 PM
gpt-3.5-turbo-16k-0613, 41 requests
113,567 prompt + 2,577 completion = 116,144 tokens
12:05 PM
Local time: Oct 23, 2023, 2:05 PM
gpt-3.5-turbo-16k-0613, 39 requests
104,608 prompt + 2,707 completion = 107,315 tokens
12:10 PM
Local time: Oct 23, 2023, 2:10 PM
gpt-3.5-turbo-16k-0613, 44 requests
119,383 prompt + 3,195 completion = 122,578 tokens
12:15 PM
Local time: Oct 23, 2023, 2:15 PM
gpt-3.5-turbo-16k-0613, 43 requests
121,550 prompt + 2,701 completion = 124,251 tokens
12:20 PM
Local time: Oct 23, 2023, 2:20 PM
gpt-3.5-turbo-16k-0613, 45 requests
121,242 prompt + 3,011 completion = 124,253 tokens
12:25 PM
Local time: Oct 23, 2023, 2:25 PM
gpt-3.5-turbo-16k-0613, 44 requests
114,829 prompt + 2,693 completion = 117,522 tokens
12:30 PM
Local time: Oct 23, 2023, 2:30 PM
gpt-3.5-turbo-16k-0613, 43 requests
123,343 prompt + 3,064 completion = 126,407 tokens
12:35 PM
Local time: Oct 23, 2023, 2:35 PM
gpt-3.5-turbo-16k-0613, 21 requests
55,659 prompt + 1,431 completion = 57,090 tokens
12:45 PM
Local time: Oct 23, 2023, 2:45 PM
gpt-3.5-turbo-16k-0613, 28 requests
76,151 prompt + 1,946 completion = 78,097 tokens
12:50 PM
Local time: Oct 23, 2023, 2:50 PM
gpt-3.5-turbo-16k-0613, 45 requests
126,193 prompt + 2,836 completion = 129,029 tokens
12:55 PM
Local time: Oct 23, 2023, 2:55 PM
gpt-3.5-turbo-16k-0613, 41 requests
109,734 prompt + 2,732 completion = 112,466 tokens
1:00 PM
101 requests
1:00 PM
Local time: Oct 23, 2023, 3:00 PM
gpt-3.5-turbo-16k-0613, 45 requests
119,494 prompt + 2,804 completion = 122,298 tokens
1:05 PM
Local time: Oct 23, 2023, 3:05 PM
gpt-3.5-turbo-16k-0613, 43 requests
121,197 prompt + 3,009 completion = 124,206 tokens
1:10 PM
Local time: Oct 23, 2023, 3:10 PM
gpt-3.5-turbo-16k-0613, 13 requests
35,452 prompt + 964 completion = 36,416 tokens
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




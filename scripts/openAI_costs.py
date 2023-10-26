#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:10:03 2023

@author: dgaio
"""

import re

text = """
12:15 PM
Local time: Oct 26, 2023, 2:15 PM
gpt-3.5-turbo-16k-0613, 43 requests
118,359 prompt + 3,056 completion = 121,415 tokens
12:20 PM
Local time: Oct 26, 2023, 2:20 PM
gpt-3.5-turbo-16k-0613, 42 requests
119,015 prompt + 2,595 completion = 121,610 tokens
12:25 PM
Local time: Oct 26, 2023, 2:25 PM
gpt-3.5-turbo-16k-0613, 29 requests
76,539 prompt + 2,025 completion = 78,564 tokens
12:30 PM
Local time: Oct 26, 2023, 2:30 PM
gpt-3.5-turbo-16k-0613, 24 requests
66,301 prompt + 1,813 completion = 68,114 tokens
12:35 PM
Local time: Oct 26, 2023, 2:35 PM
gpt-3.5-turbo-16k-0613, 45 requests
119,764 prompt + 2,901 completion = 122,665 tokens
12:40 PM
Local time: Oct 26, 2023, 2:40 PM
gpt-3.5-turbo-16k-0613, 41 requests
117,898 prompt + 2,857 completion = 120,755 tokens
12:45 PM
Local time: Oct 26, 2023, 2:45 PM
gpt-3.5-turbo-16k-0613, 4 requests
9,950 prompt + 264 completion = 10,214 tokens
12:55 PM
Local time: Oct 26, 2023, 2:55 PM
gpt-3.5-turbo-16k-0613, 12 requests
31,978 prompt + 863 completion = 32,841 tokens
1:00 PM
102 requests
1:00 PM
Local time: Oct 26, 2023, 3:00 PM
gpt-3.5-turbo-16k-0613, 44 requests
120,078 prompt + 2,963 completion = 123,041 tokens
1:05 PM
Local time: Oct 26, 2023, 3:05 PM
gpt-3.5-turbo-16k-0613, 41 requests
117,746 prompt + 2,857 completion = 120,603 tokens
1:10 PM
Local time: Oct 26, 2023, 3:10 PM
gpt-3.5-turbo-16k-0613, 17 requests
44,111 prompt + 1,118 completion = 45,229 tokens
2:00 PM
228 requests
2:10 PM
Local time: Oct 26, 2023, 4:10 PM
gpt-3.5-turbo-16k-0613, 26 requests
71,794 prompt + 1,953 completion = 73,747 tokens
2:15 PM
Local time: Oct 26, 2023, 4:15 PM
gpt-3.5-turbo-16k-0613, 44 requests
118,269 prompt + 2,823 completion = 121,092 tokens
2:20 PM
Local time: Oct 26, 2023, 4:20 PM
gpt-3.5-turbo-16k-0613, 42 requests
120,019 prompt + 2,972 completion = 122,991 tokens
2:25 PM
Local time: Oct 26, 2023, 4:25 PM
gpt-3.5-turbo-16k-0613, 44 requests
119,742 prompt + 3,074 completion = 122,816 tokens
2:30 PM
Local time: Oct 26, 2023, 4:30 PM
gpt-3.5-turbo-16k-0613, 44 requests
124,141 prompt + 2,700 completion = 126,841 tokens
2:35 PM
Local time: Oct 26, 2023, 4:35 PM
gpt-3.5-turbo-16k-0613, 28 requests
73,861 prompt + 1,887 completion = 75,748 tokens
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




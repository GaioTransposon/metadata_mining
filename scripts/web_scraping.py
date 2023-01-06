#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:21:16 2022

@author: dgaio
"""

from urllib.request import urlopen

url = "https://ontobee.org/"
page = urlopen(url)
page

html_bytes = page.read()
html = html_bytes.decode("utf-8")
print(html)


# import wget
# wget.download('https://ontobee.org/listTerms/ADO?format=tsv')

title_index = html.find("<a href=")
title_index


start_index = title_index + len("<title>")
start_index

end_index = html.find("<</a>")
end_index

title = html[start_index:end_index]
title



import re 

starts = re.findall("https://ontobee.org/listTerms", html,)
type(starts)


href="https://ontobee.org/listTerms/VT?format=tsv"





re.search(r'https://ontobee.org/listTerms\.(.*?)tsv', html).(1)


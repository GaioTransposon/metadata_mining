#! /usr/bin/python3

import json, re, sys

latitudeDegreeRegexes = list()
latitudeMinuteRegexes = list()
latitudeMinuteSecondRegexes = list()

longitudeDegreeRegexes = list()
longitudeMinuteRegexes = list()
longitudeMinuteSecondRegexes = list()

combinedDegreeRegexes = list()
combinedMinuteRegexes = list()
combinedMinuteSecondRegexes = list()

emptyRegexes = list()

match = re.compile(r"\A([\-\+]*[.,\d]*\d+)°*\s*([NS]*)[;\s,_]+([\-\+]*\d+[.,\d]*)°*\s*([WE]*)$")
combinedDegreeRegexes.append(match)
match = re.compile(r"\A([.,\d]*\d+)([NS]*)-(\d+[.,\d]*)([WE]*)$")
combinedDegreeRegexes.append(match)
match = re.compile(r"\A([.,\d]*\d+)([NS]*)_(\d+[.,\d]*)([WE]*)$")
combinedDegreeRegexes.append(match)

match = re.compile(r"\A(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]*\s*([NS]+)[\s,_]+(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]*\s*([EW]+)$")
combinedMinuteRegexes.append(match)

match = re.compile(r"\A(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]+\s*(\d+[.\d]*)[\"'`′]*\s*([NS]+)[\s,_]+(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]+\s*(\d+[.\d]*)[\"'`′]*\s*([EW]+)$")
combinedMinuteSecondRegexes.append(match)

match = re.compile(r"\A([\-\+]*\d+[.,\d]*)[º°�Â¡Á]*\s*([NS]*)$")
latitudeDegreeRegexes.append(match)

match = re.compile(r"\A(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]*\s*([NS]+)$")
latitudeMinuteRegexes.append(match)

match = re.compile(r"\A(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]+\s*(\d+[.\d]*)[\"'`′]*\s*([NS]+)$")
latitudeMinuteSecondRegexes.append(match)

match = re.compile(r"\A([\-\+]*\d+[.,\d]*)[º°�Â¡Á]*\s*([EW]*)$")
longitudeDegreeRegexes.append(match)

match = re.compile(r"\A(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]*\s*([EW]+)$")
longitudeMinuteRegexes.append(match)

match = re.compile(r"\A(\d+)[º°�Â¡Á]+\s*(\d+[.\d]*)['`′]+\s*(\d+[.\d]*)[\"'`′]*\s*([EW]+)$")
longitudeMinuteSecondRegexes.append(match)

empty = re.compile(f"\ANA$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[mM]issing$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[uU]ndefine$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[Nn]ot\s+[cC]ollected$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[Nn]ot\s+[pP]rovided$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[nN]ot\s+[aA]?pplic*l*able$")
emptyRegexes.append(empty)
empty = re.compile(f"\ANOT APPLIC*L*ABLE$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[nN]ot\s+[aA]vailable$")
emptyRegexes.append(empty)
empty = re.compile(f"\A-$")
emptyRegexes.append(empty)
empty = re.compile(f"\AN\.A\.$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[nN]\/[aA]$")
emptyRegexes.append(empty)
empty = re.compile(f"\Ana$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[nN]one$")
emptyRegexes.append(empty)
empty = re.compile(f"\A[uU]nknown$")
emptyRegexes.append(empty)

southMatchInLabel = re.compile(f"\A.*[\s\(\[]+S[\s\)\]]+.*$")
westMatchInLabel = re.compile(f"\A.*[\s\(\[]+W[\s\)\]]+.*$")

combinedMatchCounter = 0
longitudeMatchCounter = 0
latitudeMatchCounter = 0
emptyCounter = 0
failureCounter = 0

latitudeOfSample = dict()
longitudeOfSample = dict()
successfulLatRegexOfSample = dict()
successfulLonRegexOfSample = dict()
matchedLatValueOfSample = dict()
matchedLonValueOfSample = dict()

print(f" == parse_lat_lon_from_metadata.py ==", file = sys.stderr)

print(f"learning about samples that have an accidental lat-lon reversal ...", end = "", file = sys.stderr)

samplesWithLatLonReversal = set()

with open ("samples_with_lat_lon_reversal.tsv") as inFile:
	for line in inFile:
		components = line.strip().split()
		sample = components[0]
		samplesWithLatLonReversal.add(sample)

print(f"done.", file = sys.stderr)

with open ("/map/cvm_data/temp/metadata.out") as inFile:
	for line in inFile:
		components = line.strip().split(', meta {')
		jsonData = "{" + components[1]
		sample = components[0].split()[1]
		metaData = json.loads(jsonData)
		for key in metaData:

			## combined lat/lon matches

			if 'lat' in key and 'lon' in key:
				value = metaData[key]
				valueIsValid = True
				if value.count(',') > 3: valueIsValid = False
				matchFound = False
				emptyFound = False
				lat = None
				lon = None
				if valueIsValid:
					for matchRegex in combinedDegreeRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								lat = float(match[1].replace(',', '.'))
								if 'S' in match[2]: lat *= -1.0
								lon = float(match[3].replace(',', '.'))
								if 'W' in match[4]: lon *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[3]}' --")
								continue
							print(f"   COMBINED MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[3]}' '{match[4]}' -- latlon {lat}/{lon} --")
							matchFound = True
							combinedMatchCounter += 1
							matchedLatValueOfSample[sample] = value
							matchedLonValueOfSample[sample] = value
							successfulLatRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							successfulLonRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for matchRegex in combinedMinuteRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								deg = float(match[1].replace(',', '.'))
								min = float(match[2].replace(',', '.'))
								lat = deg + min / 60.0
								if 'S' in match[3]: lat *= -1.0
								deg = float(match[4].replace(',', '.'))
								min = float(match[5].replace(',', '.'))
								lon = deg + min / 60.0
								if 'W' in match[6]: lon *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[4]}' '{match[5]}'--")
								continue
							print(f"   COMBINED MINUTE MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[3]}' '{match[4]}' '{match[5]}' '{match[6]}' -- latlon {lat}/{lon} --")
							matchFound = True
							combinedMatchCounter += 1
							matchedLatValueOfSample[sample] = value
							matchedLonValueOfSample[sample] = value
							successfulLatRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							successfulLonRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for matchRegex in combinedMinuteSecondRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								deg = float(match[1].replace(',', '.'))
								min = float(match[2].replace(',', '.'))
								sec = float(match[3].replace(',', '.'))
								lat = deg + min / 60.0 + sec / 3600.0
								if 'S' in match[4]: lat *= -1.0
								deg = float(match[5].replace(',', '.'))
								min = float(match[6].replace(',', '.'))
								sec = float(match[7].replace(',', '.'))
								lon = deg + min / 60.0 + sec / 3600.0
								if 'W' in match[8]: lon *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[3]}' '{match[5]}' '{match[6]}' '{match[7]}' --")
								continue
							print(f"   COMBINED MINUTE SECOND MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[3]}' '{match[4]}' '{match[5]}' '{match[6]}' '{match[7]}' '{match[8]}' -- latlon {lat}/{lon} --")
							matchFound = True
							combinedMatchCounter += 1
							matchedLatValueOfSample[sample] = value
							matchedLonValueOfSample[sample] = value
							successfulLatRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							successfulLonRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for emptyRegex in emptyRegexes:
						match = emptyRegex.match(value)
						if match:
							print(f"   COMBINED EMPTY: -- {sample} -- {key} -- {value} -- ")
							emptyFound = True
							emptyCounter += 1
							break
					if matchFound:
						if not lat is None:
							if sample in latitudeOfSample:
								if not lat == latitudeOfSample[sample]:
									print(f"   WARNING: inconsistent mulitple latitudes seen in sample {sample}")
							latitudeOfSample[sample] = lat
						if not lon is None:
							if sample in longitudeOfSample:
								if not lon == longitudeOfSample[sample]:
									print(f"   WARNING: inconsistent multiple longitudes seen in sample {sample}")
							longitudeOfSample[sample] = lon
				if not matchFound and not emptyFound:
					print(f"   FAILED TO PARSE COMBINED: -- {sample} -- {key} -- {value} -- ")
					failureCounter += 1

			## latitude only

			if 'LATITU' in key.upper() and not 'LONGITU' in key.upper():
				value = metaData[key]
				valueIsValid = True
				if value.count(',') > 1: valueIsValid = False
				matchFound = False
				emptyFound = False
				lat = None
				if valueIsValid:
					for matchRegex in latitudeDegreeRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								lat = float(match[1].replace(',', '.'))
								if 'S' in match[2]: lat *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' --")
								continue
							print(f"   LATITUDE MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}/{lat}' '{match[2]}' -- lat {lat} --")
							matchFound = True
							latitudeMatchCounter += 1
							matchedLatValueOfSample[sample] = value
							successfulLatRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for matchRegex in latitudeMinuteRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								deg = float(match[1].replace(',', '.'))
								min = float(match[2].replace(',', '.'))
								lat = deg + min / 60.0
								if 'S' in match[3]: lat *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' --")
								continue
							print(f"   LATITUDE MINUTE MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}/{deg}' '{match[2]}/{min}' '{match[3]}' -- lat {lat} --")
							matchFound = True
							latitudeMatchCounter += 1
							matchedLatValueOfSample[sample] = value
							successfulLatRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for matchRegex in latitudeMinuteSecondRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								deg = float(match[1].replace(',', '.'))
								min = float(match[2].replace(',', '.'))
								sec = float(match[3].replace(',', '.'))
								lat = deg + min / 60.0 + sec / 3600.0
								if 'S' in match[4]: lat *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[3]}'--")
								continue
							print(f"   LATITUDE MINUTE SECOND MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}/{deg}' '{match[2]}/{min}' '{match[3]}/{sec}' '{match[4]}' -- lat {lat} --")
							matchFound = True
							latitudeMatchCounter += 1
							matchedLatValueOfSample[sample] = value
							successfulLatRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for emptyRegex in emptyRegexes:
						match = emptyRegex.match(value)
						if match:
							print(f"   LATITUDE EMPTY: -- {sample} -- {key} -- {value} -- ")
							emptyFound = True
							emptyCounter += 1
							break
					if matchFound:
						if not lat is None:
							if southMatchInLabel.match(key) and lat > 0.0:
								lat *= -1.0
							if sample in latitudeOfSample:
								if not lat == latitudeOfSample[sample]:
									print(f"   WARNING: inconsistent, multiple latitudes seen in sample {sample}, key {key}")
							latitudeOfSample[sample] = lat

				if not matchFound and not emptyFound:
					print(f"   FAILED TO PARSE LATITUDE: -- {sample} -- {key} -- {value} -- ")
					failureCounter += 1

			## longitude only matches

			if 'LONGITU' in key.upper() and not 'LATITU' in key.upper():
				value = metaData[key]
				valueIsValid = True
				if value.count(',') > 1: valueIsValid = False
				matchFound = False
				emptyFound = False
				lon = None
				if valueIsValid:
					for matchRegex in longitudeDegreeRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								lon = float(match[1].replace(',', '.'))
								if 'W' in match[2]: lon *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- {match[1]} --")
								continue
							print(f"   LONGITUDE MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}/{lon}' '{match[2]}' -- lon {lon} --")
							matchFound = True
							longitudeMatchCounter += 1
							matchedLonValueOfSample[sample] = value
							successfulLonRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for matchRegex in longitudeMinuteRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								deg = float(match[1].replace(',', '.'))
								min = float(match[2].replace(',', '.'))
								lon = deg + min / 60.0
								if 'W' in match[3]: lon *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} --")
								continue
							print(f"   LONGITUDE MINUTE MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}/{deg}' '{match[2]}/{min}' '{match[3]}' -- lon {lon} --")
							matchFound = True
							longitudeMatchCounter += 1
							matchedLonValueOfSample[sample] = value
							successfulLonRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for matchRegex in longitudeMinuteSecondRegexes:
						match = matchRegex.match(value)
						if match:
							try:
								deg = float(match[1].replace(',', '.'))
								min = float(match[2].replace(',', '.'))
								sec = float(match[3].replace(',', '.'))
								lon = deg + min / 60.0 + sec / 3600.0
								if 'W' in match[4]: lon *= -1.0
							except:
								print(f"   FAILED FLOAT CONVERSION: -- {sample} -- {key} -- {value} -- '{match[1]}' '{match[2]}' '{match[3]}'--")
								continue
							print(f"   LONGITUDE MINUTE SECOND MATCH: -- {sample} -- {key} -- {value} -- '{match[1]}/{deg}' '{match[2]}/{min}' '{match[3]}/{sec}' '{match[4]}' -- lon {lon} --")
							matchFound = True
							longitudeMatchCounter += 1
							matchedLonValueOfSample[sample] = value
							successfulLonRegexOfSample[sample] = str(matchRegex).replace('\\\\','\\')
							break
					for emptyRegex in emptyRegexes:
						match = emptyRegex.match(value)
						if match:
							print(f"   LONGITUDE EMPTY: -- {sample} -- {key} -- {value} --")
							emptyFound = True
							emptyCounter += 1
							break
					if matchFound:
						if not lon is None:
							if westMatchInLabel.match(key) and lon > 0.0:
								lon *= -1.0
							if sample in longitudeOfSample:
								if not lon == longitudeOfSample[sample]:
									print(f"   WARNING: inconsistent, multiple longitudes seen in sample {sample}")
							longitudeOfSample[sample] = lon

				if not matchFound and not emptyFound:
					print(f"   FAILED TO PARSE LONGITUDE: -- {sample} -- {key} -- {value} --")
					failureCounter += 1


print(f"{combinedMatchCounter} combined matches, {latitudeMatchCounter} latitude matches, {longitudeMatchCounter} longitude matches, {emptyCounter} empties, {failureCounter} failures")
print(f"{len(latitudeOfSample)} samples have a latitude, {len(longitudeOfSample)} samples have a longitude")

for sample in latitudeOfSample:
	if not sample in longitudeOfSample: continue
	latRegex = successfulLatRegexOfSample[sample] if sample in successfulLatRegexOfSample else "none"
	lonRegex = successfulLonRegexOfSample[sample] if sample in successfulLonRegexOfSample else "none"
	latValue = matchedLatValueOfSample[sample] if sample in matchedLatValueOfSample else "none"
	lonValue = matchedLonValueOfSample[sample] if sample in matchedLonValueOfSample else "none"
	latitudeToReport = latitudeOfSample[sample]
	longitudeToReport = longitudeOfSample[sample]
	if sample in samplesWithLatLonReversal:
		latitudeToReport = longitudeOfSample[sample]
		longitudeToReport = latitudeOfSample[sample]
	print(f"OUTPUT:\tsample\t{sample}\t{latitudeToReport}\t{longitudeToReport}\t{latValue}\t{latRegex}\t{lonValue}\t{lonRegex}")

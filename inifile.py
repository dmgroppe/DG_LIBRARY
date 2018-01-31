def inifile(fileName,operation):
	"""
	inifile creates, reads, or writes data from/to a standard ini (ascii)
       file. Such a file is organized into sections
      ([section name]), subsections(enclosed by {subsection name}),
      and keys (key=value).  Empty lines and lines with the first non-empty
      character being ; (comment lines) are ignored.
     
     Usage:
     	[keys,sections,subsections] = INIFILE(fName,'readall')
 			Reads entire file and returns all the sections, subsections
 			and keys found.
	"""

	if operation == 'readall': 
		keys,sections,subsections = ReadAllKeys(fileName) 
		return (keys,sections,subsections) 
	else: 
		raise Exception('Invalid operation passed to Inifile')

def ReadAllKeys(fileName):
	"""Read all keys out as well as sections and subsections"""
	
	try:
		fileHandle = open(fileName,'r')
	except:
		raise Exception('Error in open: file not found') 

	keys = []
	subsections = []
	sections = []
	numKeys = 0
	numSubsections = 0
	numSections = 0
	subsectionStr = ''
	sectionStr = ''
	for line in fileHandle:
		status,value,key = ProcessIniLine(line)
		if status == 1:
			numSections += 1
			sectionStr = value
			sections.append(sectionStr)
		elif status == 2:
			numSubsections += 1
			subsectionStr = value
			subsections.append([sectionStr,subsectionStr])
		elif status == 3:
			numKeys += 1
			keys.append([sectionStr,subsectionStr,key,value])
	fileHandle.close()
	return (keys,sections,subsections)


def ProcessIniLine(line):
	"""Processes a line read from the ini file and
		returns the following values:
		- status:  	-1   => unknown string found
		        	0   => empty line found
               		1   => section found
            		2   => subsection found
              		3   => key-value pair found
             		4   => comment line found (starting with ;)
   		- value:    value-string of a key, section, subsection, comment, or unknown string
   		- key:      key as string
   	"""
	status = 0
	key = '' # default; only return value possibly not set
	line = line.strip() # remove leading and trailing spaces
	endIdx = len(line)-1
	if not line: # empty sequence evaluates to false
		status = 0
		key = ''
		value = ''
		return (status,key,value)
	elif line[0]==';': # comment found
		status = 4
		value = line[1:endIdx+1]
	elif (line[0] == '[') and (line[endIdx] == ']') and (endIdx+1 >= 3): # section found
		status = 1
		value = line[1:endIdx].lower()
	elif (line[0] == '{') and (line[endIdx] == '}') and (endIdx+1 >= 3): # subsection found
		status = 2
		value = line[1:endIdx].lower()
	else: # key found
		try:
			pos = line.index('=')
		except: # '=' not found in line
			status = -1
			value = line
			return (status,value,key)
		status = 3
		key = line[0:pos].lower()
		key.strip()
		if not key:
			satus = -1
			key = ''
			value = ''
		else:
			value = line[pos+1:endIdx+1].lower()
			value.strip()
	return (status,value,key)
# by БорискинМА (13.09.19), 3341506/90401
# GNU nano 2.9.3, python3
# Task 2
file = open("SMSSpamCollection.txt", "r", encoding='utf-8')

ham = 0
spam = 0

def createFiles(fileName, line):
	f = open(fileName, "w", encoding='utf-8')
	f.write(line)
	f.close()
	f = open(fileName, "r", encoding='utf-8')
	str = f.read().split("\t")
	str[0] = ""
	str = "".join(str)
	f.close()
	f = open(fileName, "w", encoding='utf-8')
	f.write(str)
	f.close()

for line in file:
	if "ham" in line:
		ham += 1
		fileName = "ham/" + str(ham) + ".txt"
		createFiles(fileName, line)

	elif "spam" in line:
		spam += 1
		fileName = "spam/" + str(spam) + ".txt"
		createFiles(fileName, line)

file.close()
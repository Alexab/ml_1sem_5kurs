# by БорискинМА (13.09.19), 3341506/90401
# GNU nano 2.9.3, python3
# Task 2
file = open("SMSSpamCollection.txt", "r")

ham = 0
spam = 0

def rmWords(fileName, line):
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
		ham = ham+1
		rmWords("ham/" + str(ham) + ".txt", line)

	elif "spam" in line:
		spam = spam+1
		rmWords("spam/" + str(spam) + ".txt", line)

file.close()

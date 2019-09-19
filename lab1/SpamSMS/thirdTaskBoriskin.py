# by БорискинМА (19.09.19), 3341506/90401
# Sublime Text, python3
# Tasks 3
import secondTaskBoriskin

ham = secondTaskBoriskin.ham
spam = secondTaskBoriskin.spam

def symbolsCounter(fileName):
	f = open(fileName, "r", encoding='utf-8')
	for line in f:
		symbols = len(line)
	f.close()
	return symbols

def minMaxInFiles(folder, N):
	totalSymbols = 0
	minLen = 0
	maxLen = 0
	aboutLines = (0,0,0,0)

	for i in range(N):
		fileName = folder + str(i+1) + ".txt"
		filePrevName = folder + str(i) + ".txt"
		if i > 1:
			if symbolsCounter(fileName) < symbolsCounter(filePrevName):
				minLen = symbolsCounter(fileName)
			elif symbolsCounter(fileName) > symbolsCounter(filePrevName):
				maxLen = symbolsCounter(fileName)
		else:
			length = (symbolsCounter(fileName), symbolsCounter(fileName))

		totalSymbols += symbolsCounter(fileName)

		if folder == "spam/":
			aboutLines = (minLen,maxLen,0,totalSymbols)
		else:
			aboutLines = (minLen,maxLen,totalSymbols,0)

	return aboutLines


totalHamLength = minMaxInFiles("ham/", ham) 
totalSpamLength = minMaxInFiles("spam/", spam)

if totalHamLength[0] < totalSpamLength[0]:
	minLen = totalHamLength [0]
else:
	minLen = totalSpamLength[0]

if totalHamLength[1] > totalSpamLength[1]:
	maxLen = totalHamLength[1]
else:
	maxLen = totalSpamLength[1]

print("Минимальная длина содержимого ham файлов:", totalHamLength[0])
print("Максимальная длина содержимого ham файлов:", totalHamLength[1])
print("Средняя длина содержимого ham файлов:", totalHamLength[2]/ham)

print("\nМинимальная длина содержимого spam файлов:", totalSpamLength[0])
print("Максимальная длина содержимого spam файлов:", totalSpamLength[1])
print("Средняя длина содержимого spam файлов:", totalSpamLength[3]/spam)

print("\nМинимальная длина содержимого всех файлов:", minLen)
print("Максимальная длина содержимого всех файлов:", maxLen)
print("Средняя длина содержимого всех файлов:", (totalHamLength[2]+totalSpamLength[3])/(ham+spam))


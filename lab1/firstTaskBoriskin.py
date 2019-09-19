# by БорискинМА (13.09.19), 3341506/90401
# GNU nano 2.9.3, python3
# Task 1
import random
# первое задание
def arithmetic (x,y,o):
	if o == "+":
		return x+y
	elif o == "-":
		return x-y
	elif o == "*":
		return x*y
	elif o == "/":
		return x/y
	else:
		return "Неизвестная операция"

operations=["+","-","*","/"]
for i in operations:
        print("для x=15,y=5 операция",i,":",arithmetic(15,5,i))

#второе задание
def is_year_leap(y):
	if y % 4 != 0 or (y % 100 == 0 and year %400 != 0):
		return False
	else:
		return True

years=[2019,2020,2021,2022,2023,2024,2025]
for i in years:
	print("високосный",i,":",is_year_leap(i))

#третье задание
def square(a):
	return (a*4,a*a,(2*(a**2))**(1/2))

for i in range (3):
        a = random.randint(3,12)
        print("P, S, d для квадрата со стороной",a,":", square(a))

#четвертое задание
def season(m):
	if m == 1 or m == 2 or m == 12:
		return "зима"
	elif m == 3 or m == 4 or m == 5:
		return "весна"
	elif m == 6 or m == 7 or m == 8:
		return "лето"
	else:
		return  "осень"

for i in range (3):
	m = random.randint(1,12)
	print("месяц",m,":", season(m))

#пятое задание
def bank(a, years):
	for i in range (years):
		a = a*1.1
	return a

print("Сумма на счету будет:", bank(3000, 5))

#шестое задание
def is_prime(x):
	#у любого составного числа есть делитель,
	#не превосходящий квадратного корня из числа
	divider = 2
	while divider**2 <= x and x % divider != 0:
		divider += 1
	return divider**2 > x

number = random.randint(0,1000)
print("Наше число", number, "простое?", is_prime(number))

#седьмое задание

def monthCheck(mm):
	return mm > 0 and mm <= 12

def dayCheck(dd, mm):
	dd_in_mm = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
	return dd > 0 and dd_in_mm[mm]

def yearCheck(yy):
	return yy >= 0 and yy <= 5000000000

def date(dd, mm, yy):
	return dayCheck(dd, mm) and monthCheck(mm) and yearCheck(yy)

today = [random.randint(1,31), random.randint(1,12), random.randint(1997,2097)]
print ("Сегодня", today[0], "/", today[1], "/", today[2], "- валидно?", date(today[0],today[1],today[2]))

#восьмое задание
def XOR_cipher(str_k, key):
	crypt = ""
	for i in str_k:
		# A = 65 ASCII, B = 66, C = 67 ...
		crypt += chr(ord(i)^len(key))
	return crypt

str_k = "В Берне идут переговоры между генералом Вольфом и американской стороной"
key = "Юстас"
print("строка:", str_k, "/// ключ:", key, "/// шифровка:", XOR_cipher(str_k,key))

def XOR_uncipher(str_unk, key):
	decrypt = ""
	for i in str_unk:
		decrypt += chr(ord(i)^len(key))
	return decrypt

print("шифровка:", XOR_cipher(str_k,key), "/// ключ:", key, "/// дешифровка:", XOR_uncipher(XOR_cipher(str_k,key), key))

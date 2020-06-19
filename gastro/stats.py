import os
count = [0,0,0,0,0,0,0,0,0,0] 
for filename in os.listdir('D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/Gastro_dataset/Labels'):
	if filename.endswith('txt'):
		with open(os.path.join('D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/Gastro_dataset/Labels',filename), 'r') as file:
			lines = file.readlines()
			for line in lines:
				count[int(line[0])] =  count[int(line[0])] + 1
        	
print(count)     	


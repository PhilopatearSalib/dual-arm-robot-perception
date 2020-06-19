import os
f= open('D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/Gastro_dataset/train.txt','w')
for filename in os.listdir('D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/Gastro_dataset/images'):
    f.write('D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/Gastro_dataset/images'+filename+'\n')
	

import glob
import math
from random import shuffle
from nltk.corpus import stopwords

training_neg = 810
training_pos = 810

validation_neg = 900
validation_pos = 900

list_of_neg_files = glob.glob('review_polarity/txt_sentoken/neg/*.txt')
list_of_pos_files = glob.glob('review_polarity/txt_sentoken/pos/*.txt')
shuffle(list_of_neg_files)
shuffle(list_of_pos_files)

#test_neg = len(list_of_neg_files)
#test_pos = len(list_of_pos_files)

test_neg = 1000
test_pos = 1000

print(str(training_neg) + " " + str(validation_neg) + " " + str(test_neg))

class Review:
	def __init__(self):
		self.file_name = ''
		self.bag_of_words = []
		self.frequency = []
		self.tag = 0


	def read_data(self):
		input_file = open(self.file_name, 'r')
		temp_review = Review()
		lines = input_file.readlines();
		all_words = []
		useless_word = set(stopwords.words('english'))
		for line in lines:
			word = ''
			for i in range(0, len(line)):
				if ((line[i] == ' ') or (line[i] == '\n')):
					if(len(word) > 2):
						if not (word in useless_word):
							if not any(ch.isdigit() for ch in word):
								all_words.append(word)
								if word not in self.bag_of_words:
									self.bag_of_words.append(word)
					word = ''
				else:
					word = word + line[i]
		for i in range(0, len(self.bag_of_words)):
			self.frequency.append(all_words.count(self.bag_of_words[i]))
		input_file.close()

class Collection:
	def __init__(self):
		self.review = []

	def read_data(self, file_name, tag):
		temp_review = Review()
		temp_review.file_name = file_name
		temp_review.tag = tag
		temp_review.read_data()
		self.review.append(temp_review)

def l1norm(bag_of_words1, frequency1, bag_of_words2, frequency2):
	final_list = list(set().union(bag_of_words1, bag_of_words2))
	distance = 0
	for word in final_list:
		a = 0
		b = 0
		if word in bag_of_words1:
			a = frequency1[bag_of_words1.index(word)]
		if word in bag_of_words2:
			b = frequency2[bag_of_words2.index(word)]
		distance = distance + abs(a - b)
	return distance

def l2norm(bag_of_words1, frequency1, bag_of_words2, frequency2):
	final_list = list(set().union(bag_of_words1, bag_of_words2))
	distance = 0
	for word in final_list:
		a = 0
		b = 0
		if word in bag_of_words1:
			a = frequency1[bag_of_words1.index(word)]
		if word in bag_of_words2:
			b = frequency2[bag_of_words2.index(word)]
		distance = distance + ((a - b) * (a - b))
	return math.sqrt(distance)

def linfinitynorm(bag_of_words1, frequency1, bag_of_words2, frequency2):
	final_list = list(set().union(bag_of_words1, bag_of_words2))
	distance = 0
	for word in final_list:
		a = 0
		b = 0
		if word in bag_of_words1:
			a = frequency1[bag_of_words1.index(word)]
		if word in bag_of_words2:
			b = frequency2[bag_of_words2.index(word)]
		if (distance < abs(a - b)):
			distance = abs(a - b)
	return distance

def return_tag(distance_tag, k):
	count_0 = 0
	count_1 = 0
	for i in range(0, k):
		if(distance_tag[i][1] == 0):
			count_0 = count_0 + 1;
		elif(distance_tag[i][1] == 1):
			count_1 = count_1 + 1;
	if(count_1 > count_0):
		return 1
	return 0

training_collection = Collection()
validation_collection = Collection()
test_collection = Collection()

for i in range(0, training_neg):
	training_collection.read_data(list_of_neg_files[i],0)

for i in range(training_neg, validation_neg):
	validation_collection.read_data(list_of_neg_files[i], 0)

for i in range(validation_neg, test_neg):
	test_collection.read_data(list_of_neg_files[i], 0)

for i in range(0, training_pos):
	training_collection.read_data(list_of_pos_files[i],1)

for i in range(training_pos, validation_pos):
	validation_collection.read_data(list_of_pos_files[i], 1)

for i in range(validation_pos, test_pos):
	test_collection.read_data(list_of_pos_files[i], 1)
	
min_error = 1.0
min_l = 0
min_k = 0

#for l = 1, for all k
error1 = 0.0
error2 = 0.0
error3 = 0.0
for x in validation_collection.review:
	distance_tag = []
	for y in training_collection.review:
		temp = l1norm(x.bag_of_words, x.frequency, y.bag_of_words, y.frequency)
		distance_tag.append([temp, y.tag])
	distance_tag.sort(key = lambda x : x[0])
	min_tag1 = return_tag(distance_tag, 1)
	min_tag2 = return_tag(distance_tag, 3)
	min_tag3 = return_tag(distance_tag, 5)
	if(x.tag != min_tag1):
		error1 = error1 + 1.0
	if(x.tag != min_tag2):
		error2 = error2 + 1.0
	if(x.tag != min_tag3):
		error3 = error3 + 1.0
error1 = error1 / len(validation_collection.review)
error2 = error2 / len(validation_collection.review)
error3 = error3 / len(validation_collection.review)
print("Error for l = 1, k = 1 : " + str(error1))
if(min_error > error1):
	min_error = error1
	min_l = 1
	min_k = 1
print("Error for l = 1, k = 3 : " + str(error2))
if(min_error > error2):
	min_error = error2
	min_l = 1
	min_k = 3
print("Error for l = 1, k = 5 : " + str(error3))
if(min_error > error3):
	min_error = error3
	min_l = 1
	min_k = 5

#for l = 2, for all k
error1 = 0.0
error2 = 0.0
error3 = 0.0
for x in validation_collection.review:
	distance_tag = []
	for y in training_collection.review:
		temp = l2norm(x.bag_of_words, x.frequency, y.bag_of_words, y.frequency)
		distance_tag.append([temp, y.tag])
	distance_tag.sort(key = lambda x : x[0])
	min_tag1 = return_tag(distance_tag, 1)
	min_tag2 = return_tag(distance_tag, 3)
	min_tag3 = return_tag(distance_tag, 5)
	if(x.tag != min_tag1):
		error1 = error1 + 1.0
	if(x.tag != min_tag2):
		error2 = error2 + 1.0
	if(x.tag != min_tag3):
		error3 = error3 + 1.0
error1 = error1 / len(validation_collection.review)
error2 = error2 / len(validation_collection.review)
error3 = error3 / len(validation_collection.review)
print("Error for l = 2, k = 1 : " + str(error1))
if(min_error > error1):
	min_error = error1
	min_l = 2
	min_k = 1
print("Error for l = 2, k = 3 : " + str(error2))
if(min_error > error2):
	min_error = error2
	min_l = 2
	min_k = 3
print("Error for l = 2, k = 5 : " + str(error3))
if(min_error > error3):
	min_error = error3
	min_l = 2
	min_k = 5

#for l = infinity, for all k
error1 = 0.0
error2 = 0.0
error3 = 0.0
for x in validation_collection.review:
	distance_tag = []
	for y in training_collection.review:
		temp = linfinitynorm(x.bag_of_words, x.frequency, y.bag_of_words, y.frequency)
		distance_tag.append([temp, y.tag])
	distance_tag.sort(key = lambda x : x[0])
	min_tag1 = return_tag(distance_tag, 1)
	min_tag2 = return_tag(distance_tag, 3)
	min_tag3 = return_tag(distance_tag, 5)
	if(x.tag != min_tag1):
		error1 = error1 + 1.0
	if(x.tag != min_tag2):
		error2 = error2 + 1.0
	if(x.tag != min_tag3):
		error3 = error3 + 1.0
error1 = error1 / len(validation_collection.review)
error2 = error2 / len(validation_collection.review)
error3 = error3 / len(validation_collection.review)
print("Error for l = infinity, k = 1 : " + str(error1))
if(min_error > error1):
	min_error = error1
	min_l = 3
	min_k = 1
print("Error for l = infinity, k = 3 : " + str(error2))
if(min_error > error2):
	min_error = error2
	min_l = 3
	min_k = 3
print("Error for l = infinity, k = 5 : " + str(error3))
if(min_error > error3):
	min_error = error3
	min_l = 3
	min_k = 5

print("Final error : " + str(min_error) + " l = " + str(min_l) + " k = " + str(min_k))

final_error = 0.0
for x in test_collection.review:
	distance_tag = []
	for y in training_collection.review:
		if(min_l == 1):
			temp = l1norm(x.bag_of_words, x.frequency, y.bag_of_words, y.frequency)
			distance_tag.append([temp, y.tag])
		elif(min_l == 2):
			temp = l2norm(x.bag_of_words, x.frequency, y.bag_of_words, y.frequency)
			distance_tag.append([temp, y.tag])
		elif(min_l == 3):
			temp = linfinitynorm(x.bag_of_words, x.frequency, y.bag_of_words, y.frequency)
			distance_tag.append([temp, y.tag])
	distance_tag.sort(key = lambda x : x[0])
	min_tag = return_tag(distance_tag, min_k)
	if(x.tag != min_tag):
		final_error = final_error + 1.0
final_error = final_error / len(test_collection.review)
print("Test error : " + str(final_error) + " l = " + str(min_l) + " k = " + str(min_k))
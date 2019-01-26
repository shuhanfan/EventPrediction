import pandas as pd
from alphabet import Alphabet
import os
def generate_instance_Ids(input_path, train_files, test_files, event_alphabet, window_size, interval_size):
	# get the train data
	train_data = []
	for train_day in train_files:
		train_path = input_path + "/" + train_day
		files= os.listdir(train_path) #得到文件夹下的所有文件名称
		for file in files: #遍历mid文件夹
			#print("[generate_instance_Ids]==> input train file is:",(train_path+"/"+file))
			try:
				df = pd.read_csv(train_path+"/"+file, encoding = 'utf-8', delimiter="\t", usecols=[0], error_bad_lines=False)
			except pd.errors.ParserError:
				df = pd.read_csv(train_path+"/"+file, encoding = 'utf-8', engine='python', delimiter="\t", usecols=[0], error_bad_lines=False)
				
			eventid = df["newEID"]
			for i in range(1, len(eventid)-window_size, interval_size):
				train_data.append((event_alphabet.get_index_list(eventid[i-1:i-1+window_size-1]),event_alphabet.get_index_list(eventid[i:i+window_size-1])))
	# get the test data
	test_data = []
	for test_day in test_files:
		test_path = input_path + "/" + test_day
		files= os.listdir(test_path) #得到文件夹下的所有文件名称
		for file in files: #遍历mid文件夹
			try:
				df = pd.read_csv(test_path+"/"+file, encoding = 'utf-8', delimiter="\t", usecols=[0], error_bad_lines=False)
			except Exception as e:
				df = pd.read_csv(test_path+"/"+file, encoding = 'utf-8', engine='python', delimiter="\t", usecols=[0], error_bad_lines=False)

			eventid = df["newEID"]
			for i in range(1, len(eventid)-window_size, interval_size):
				test_data.append((event_alphabet.get_index_list(eventid[i-1:i-1+window_size-1]),event_alphabet.get_index_list(eventid[i:i+window_size-1])))
	return train_data, test_data
	



def build_alphabet(input_file, train_files, test_files):
	event_alphabet = Alphabet("eventid")
	# deal with the train file
	for train_day in train_files:
		train_path = input_file + "/" + train_day
		files= os.listdir(train_path) #得到文件夹下的所有文件名称
		for file in files: #遍历mid文件夹
			in_lines = open(train_path+"/"+file, 'r', encoding='utf-8').readlines()
			for idx in range(len(in_lines)):
				eventid = in_lines[idx].split('\t')[0]
				event_alphabet.add(eventid)
	# deal with the test file
	for test_day in test_files:
		test_path = input_file + "/" + test_day
		files= os.listdir(test_path) #得到文件夹下的所有文件名称
		for file in files: #遍历mid文件夹
			in_lines = open(test_path+"/"+file, 'r', encoding='utf-8').readlines()
			for idx in range(len(in_lines)):
				eventid = in_lines[idx].split('\t')[0]
				event_alphabet.add(eventid)
	
	return event_alphabet



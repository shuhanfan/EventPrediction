import time
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
import data
from lstm_network import SeqModel

def calculate_loss(loss_op,logits,batch_label,model,l2_weight, device):
    classify_loss = loss_op(logits,batch_label)
    l2_loss = Variable(torch.FloatTensor([0]), requires_grad=True)
    if args.cuda:
        l2_loss= l2_loss.to(device)
    l2_list = [W.norm(2) for W in filter(lambda p: p.requires_grad,model.parameters())]
    l2_loss = sum(l2_list)
    loss=classify_loss+l2_weight*l2_loss
    return loss

def batchify_with_label(input_batch_list, cuda, device):
	# print("input_batch_list\n", input_batch_list)
	eventid = [data[0] for data in input_batch_list]
	label = [data[1] for data in input_batch_list]
	# print("\neventid\n",eventid)
	# print("\label\n",label)
	#print("eventid,",eventid)
	eventid = torch.LongTensor(eventid)
	label = torch.LongTensor(label)
	#label = torch.LongTensor(label)
	#print("\neventid\n",eventid)
	#label = Variable(torch.LongTensor(label), requires_grad=False)
	if cuda:
		# print("cuda")
		eventid = eventid.to(device)
		label = label.to(device)
		# print("after cuda")
	return eventid, label

def train(train_data, model, args, optimizer, loss_op, device, epoch):
	#print("[train]==>", train_data)
	accumulate_loss = []
	accumulate_pred = []
	accumulate_gold = []
	start_time = time.time()
	# set model in train mode
	model.train()
	batch_size = args.batch_size
	train_num = len(train_data)
	total_batch = train_num//batch_size+1
	for batch_id in range(total_batch):
		model.train()
		start = batch_id * batch_size
		end = (batch_id + 1) * batch_size
		if (end > train_num): # 结尾先截掉
			break
		instance = train_data[start:end]
		if(not instance):
			continue
		batch_eventid, batch_label = batchify_with_label(instance, args.cuda, device)
		logits = model(batch_eventid)
		# print(logits.shape) #[batch_size*seq_len]
		predict = torch.max(logits, 1)[1]
		#print("[train]==> predict:",predict)
		batch_label = batch_label.view([-1])
		#print("[train]==> batch_label:", batch_label)
		loss= calculate_loss(loss_op,logits,batch_label,model,args.l2_weight, device)
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()#根据参数的当前梯度更新参数
		accumulate_gold.extend(batch_label.tolist())
		accumulate_pred.extend(predict.tolist())
		accumulate_loss.append(loss.cpu().data.numpy())
	if epoch % args.display_epoch == 0 and batch_id > 0:
		ret=util.evaluate_result(accumulate_pred, accumulate_gold)
		elapsed = time.time() - start_time
		#print("the accumulate_loss is:", accumulate_loss)
		print('mode:train | epoch {:3d} |  ms/batch {:5.2f} | loss {:5.4f} | ret {}'.format(
              epoch, elapsed * 1000, np.mean(accumulate_loss), ret))
		start_time = time.time()

def test(test_data, model, args, loss_op, device, epoch):
	accumulate_loss = []
	accumulate_pred = []
	accumulate_gold = []
	start_time = time.time()
	# set model in train mode
	
	batch_size = args.batch_size
	train_num = len(train_data)
	total_batch = train_num//batch_size+1
	for batch_id in range(total_batch):
		start = batch_id * batch_size
		end = (batch_id + 1) * batch_size
		if (end > train_num): # 结尾先截掉
			break
		instance = train_data[start:end]
		if(not instance):
			continue
		batch_eventid, batch_label = batchify_with_label(instance, args.cuda, device)
		logits = model(batch_eventid)
		# print("[train]==> logit:",logit)
		# print(logits.shape) #[batch_size*seq_len]
		predict = torch.max(logits, 1)[1]
		#print("[test]==> predict:",predict)
		batch_label = batch_label.view([-1])
		#print("[test]==> batch_label:", batch_label)
		loss=calculate_loss(loss_op,logits,batch_label,model,args.l2_weight,device)
		accumulate_pred.extend(predict.tolist())
		accumulate_gold.extend(batch_label.tolist())
		accumulate_loss.append(loss.cpu().data.numpy())
	if epoch % args.display_epoch == 0 and batch_id > 0:
		ret=util.evaluate_result(accumulate_pred, accumulate_gold)
		elapsed = time.time() - start_time
		print('model: test | epoch {:3d} | ms/batch {:5.2f} | loss {:5.4f} | ret {}'.format(
              epoch, elapsed * 1000 , np.mean(accumulate_loss), ret))
		start_time = time.time()


if __name__ == '__main__':
	args = util.get_args()	
	if torch.cuda.is_available():
		if not args.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	device = torch.device("cuda" if args.cuda else "cpu")

	
	event_alphabet = data.build_alphabet(args.input_file_path, args.train_file, args.test_file)
	args.category_num = event_alphabet.size()
	
	train_data, test_data = data.generate_instance_Ids(args.input_file_path, args.train_file, args.test_file, event_alphabet, args.window_size, args.interval_size)
	net = SeqModel(args, device).to(device)
	model = torch.nn.DataParallel(net, device_ids=[0,1]) # 实现dataparallel
	
	loss_op = nn.NLLLoss()# nn.CrossEntropyLoss()
	#print(model.parameters())
	#print(model.parameters)
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	#print(parameters)
	optimizer = optim.Adam(parameters, lr=args.init_learning_rate)
	#print("parameters")
	## start training
	for epoch in range(args.training_epochs):
		train(train_data, model, args, optimizer, loss_op, device, epoch)
		test(test_data, model, args, loss_op, device, epoch)
		# optimizer = util.lr_decay(optimizer, epoch, args.learning_rate_decay, init_lr=args.init_learning_rate)




import argparse
import numpy as np

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--init-learning-rate', type=float, default=0.015, 
    	help='initial learning rate')
	parser.add_argument('--learning-rate-decay', type=float, default=0.1,
                        help='initial learning rate')
	parser.add_argument('--training-epochs', type=int, default=3000,
                        help='the training step number')
	parser.add_argument('--batch-size', type=int, default=128, #512
                        help='the number of trainging event each batch')
	parser.add_argument('--gpunum', type=int, default=2, #512
                        help='the number of gpu to use')
	parser.add_argument('--display-epoch', type=int, default=1,
                        help='the epoch number to display the Precise')
	parser.add_argument('--embedding-dim',type=int,default=200,
                        help='the dimension of embedding')
	parser.add_argument('--hidden-dim',type=int,default=400,
                        help='the hidden dimension of lstm')
	parser.add_argument('--input-file-path', type=str, default='/root/DM/data/dataByDate_newid_norepeat')
	parser.add_argument('--train-file', type=str, nargs='+', default=['2018-08-01'])
	parser.add_argument('--test-file',  type=str, nargs='+', default=['2018-08-02'])

	parser.add_argument('--l2-weight', type=float, default=1e-3)
    #parser.add_argument('--candidate-num', type=int, default=6,
#                        help='the number suggest by GAN')
	parser.add_argument('--cuda', default=False, action='store_true',
                        help='whether to use gpu')
	parser.add_argument('--categery-num', type=int, default=8748,
                        help='the category of event type')
	parser.add_argument('--num-layers', type=int, default=128,
                        help='stack num_layers lstm together to form a stacked LSTM')
	parser.add_argument('--dropout', type=int, default=0,
                        help='dropout probability for the LSTM')
	parser.add_argument('--window-size', type=int, default=20,
                        help='window size for LSTM sequence')
	parser.add_argument('--interval-size', type=int, default=5,
                        help='interval gap between the previous and the latter sequence')
	parser.add_argument('--bidirectional', type=bool, default=False,
						help='whether use a bidirectional LSTM')
	return parser.parse_args()


def lr_decay(optimizer, epoch, decay_rate, init_lr):
	lr = init_lr * ((1-decay_rate)**epoch)
    # print( " Learning rate is setted as:", lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer


def evaluate_result(pred, gold, negid=-1):
	"""
        input:
            pred (all result): pred tag result, in numpy format
            gold (all result): gold result variable
    """
	gold_total, pred_total, pred_right = 0, 0, 0
	pred = np.array(pred)
	gold = np.array(gold)
	for index in range(pred.shape[0]):
		a=int(pred[index])
		b=int(gold[index])
		if a != negid: pred_total += 1
		if b != negid: gold_total += 1
		if a != negid and a == b: pred_right += 1
	ret = ('Pred_total: %d, Pred_right: %d, Gold_total: %d ' % (pred_total, pred_right, gold_total))
	precision = 0.0 if pred_total == 0 else 1. * pred_right / pred_total
	# recall = 0.0 if gold_total == 0 else 1. * pred_right / gold_total
	# f = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
	# ret += ('P:%.2f, R:%.2f, F:%.2f' % (100 * precision, 100 * recall, 100 * f))
	ret += ('P:%.2f' % ( 100*precision))
	return ret

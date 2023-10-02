


def seq_logo_reverse(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):

	def get_nt_height(pwm, height, norm):

		def entropy(p):
			s = 0
			for i in range(len(p)):
				if p[i] > 0:
					s -= p[i]*np.log2(p[i])
			return s

		num_nt, num_seq = pwm.shape
		heights = np.zeros((num_nt,num_seq));
		for i in range(num_seq):
			if norm == 1:
				total_height = height
			else:
				total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
			if alphabet == 'pu':
				heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
			else:
				heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

		return heights.astype(int)


def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):

	def get_nt_height(pwm, height, norm):

		def entropy(p):
			s = 0
			for i in range(len(p)):
				if p[i] > 0:
					s -= p[i]*np.log2(p[i])
			return s

		num_nt, num_seq = pwm.shape
		heights = np.zeros((num_nt,num_seq));
		for i in range(num_seq):
			if norm == 1:
				total_height = height
			else:
				total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
			if alphabet == 'pu':
				heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
			else:
				heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

		return heights.astype(int)

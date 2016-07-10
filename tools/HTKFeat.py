import numpy as np
import os

import pickle
from tools.autoencoder_model import *

class MFCC_HTK:
	""" Class to compute HTK compatible MFCC features from audio.

		It is designed to be as close as possible to the HTK implementation.
		For details on HTK implementationlook for HTK Book online, specifically 
		chapter 5 titled "Speech Input/Output".

		This implementation was somewhat based upon the HTK source code. Most
		of the interesting code can be found in .../HTKLib/HSigP.c file.

		The latest version of HTK available during writing of this class was 3.4.1.

		HTK is licensed by University of Cambridge and isn't in any way related
		to this particular Python implementation (so don't bother them if you 
		have problems with this code).

		For more information about HTK go to http://htk.eng.cam.ac.uk/
	"""

	win_len=400
	win_shift=160
	preemph=0.97
	filter_num=24
	lifter_num=22
	mfcc_num=12
	lo_freq = 80
	hi_freq = 7500
	samp_freq=16000
	filter_mat=None	
	raw_energy=False

	feat_melspec=False
	feat_mfcc=True
	feat_energy=True
	enc_dec=None

	def load_filter(self,filename):
		""" Internal load filter method -- you don't need to run this yourself. 
			Loads filter spec from file.
		"""

		if not os.path.isfile(filename):
			raise IOError(filename)

		data=np.genfromtxt(filename,delimiter=',')

		self.filter_num=np.asscalar(np.max(data[:,1]).astype('int'))
		self.filter_mat=np.zeros((self.fft_len/2,self.filter_num))
		
		for i in range(len(data)):

			wt=data[i,0]
			bin=np.asscalar(data[i,1].astype('int'))

			if(bin<0):
				continue

			if(bin>0):
				self.filter_mat[i,bin-1]=wt

			if(bin<self.filter_num):
				self.filter_mat[i,bin]=1-wt


	def create_filter(self, num):
		""" Internal create filter method -- you don't need to runthis yourself.
			Creates filter specified by their count.
		"""

		self.filter_num=num
		self.filter_mat=np.zeros((self.fft_len//2,self.filter_num))

		mel2freq = lambda mel: 700.0*(np.exp((mel)/1127.0)-1)
		freq2mel = lambda freq: 1127*(np.log(1+((freq)/700.0)))

		lo_mel=freq2mel(self.lo_freq);
		hi_mel=freq2mel(self.hi_freq);

		mel_c=np.linspace(lo_mel,hi_mel,self.filter_num+2)
		freq_c=mel2freq(mel_c);

		point_c=freq_c/float(self.samp_freq)*self.fft_len
		point_c=np.floor(point_c).astype('int')

		
		for f in range(self.filter_num):
			d1=point_c[f+1]-point_c[f]
			d2=point_c[f+2]-point_c[f+1]

			self.filter_mat[point_c[f]:point_c[f+1]+1,f]=np.linspace(0,1,d1+1)
			self.filter_mat[point_c[f+1]:point_c[f+2]+1,f]=np.linspace(1,0,d2+1)


	def __init__(self, 
		filter_file=None,	
		win_len=400,
		win_shift=160,
		preemph=0.97,
		filter_num=26,
		lifter_num=22,
		mfcc_num=12,
		lo_freq = 80,
		hi_freq = 7500,
		samp_freq=16000,
		raw_energy=False,
		feat_melspec=False,
		feat_mfcc=True,
		feat_energy=True,
		encoding_net='models/encoder_adam_400.dat'):
		""" Class contructor -- you can set all the processing parameters here.

			Args:
				filter_file (string): load the filter specification from a file. This exists
					to allow binary comaptibility with HTK, because they implement the filters
					slightly differently than mentioned in their docs.

					The format of filter file is CSV, where each line contains two values:
					weight and id of the filter at the given spectrum point. The number of 
					lines is equal to the number of spectral points computed by FFT (e.g. 256
					for 512-point FFT - half due to Nyquist).

					The file contains only raising edges of the filters. The falling edges are
					computed by taking the rasing edge of the next filter and inverting it (i.e
					computing 1-x). Filter id -1 means there is no filter at that point.

					If you set filter_file is None, a built-in method will be used to create
					half-overlapping triangular filters spread evenly between lo_freq and
					hi_freq in the mel domain.

				win_len (int): Length of frame in samples. Default value is 400, which is
					equal to 25 ms for a signal sampled at 16 kHz (i.e. 2.5x the win_shift length)
				
				win_shift (int): Frame shift in samples - in other words, distance between
					the start of two consecutive frames. Default value is 160, which is 
					equal to 10 ms for a signal sampled at 16 kHz. This is generates 100 frames
					per second of the audio, which is a standard framerate for many audio tasks.

				preemph (float): Preemphasis coefficient. This is used to calculate first-order
					difference of the signal.

				filter_num (int): Number of triangular filters used to reduce the spectrum. Default
					value is 24. If filter_file is used (set to different than None), this value is
					overwritten with the contents of the filter_file.

				mfcc_num (int): Number of MFCCs computed. Default value is 12.

				lo_freq (float): Lowest frequency (in Hz) used in computation. Default value is
					80 Hz. This is used exclusively to compute filters. The value is ignored if
					filters are loaded from file.

				hi_freq (float): Highest frequency (in Hz) used in computation. Default value is
					7500 Hz. This is used exclusively to compute filters. The value is ignored if
					filters are loaded from file.

				samp_freq (int): Sampling frequency of the audio. Default value is 16000, which is
					a common value for recording speech. Due to Nyquist, the maximum frequency stored
					is half of this value, i.e. 8000 Hz.

				raw_energy (boolean): Should the energy be computed from the raw signal, or (if false)
					should the 0'th coepstral coefficient be used instead, which is almost equivalent
					and much faster to compute (since we compute MFCC anyway).

				feat_melspec (boolean): Should the spectral features be added to output. These are
					the values of the logarithm of the filter outputs. The number of these features
					is eqeual to filter_num.

				feat_mfcc (boolean): Should MFCCs be added to the output. The number of these features
					is equal to mfcc_num.

				feat_energy(boolean): Should energy be added to the output. This is a single value.
		"""
		self.__dict__.update(locals())

		self.fft_len=np.asscalar(2**(np.floor(np.log2(self.win_len))+1).astype('int'))

		if filter_file:
			self.load_filter(filter_file)
		else:
			self.create_filter(filter_num)

		self.hamm=np.hamming(self.win_len)		

		self.dct_base=np.zeros((self.filter_num,self.mfcc_num));
		for m in range(self.mfcc_num):
			self.dct_base[:,m]=np.cos((m+1)*np.pi/self.filter_num*(np.arange(self.filter_num)+0.5))

		self.lifter=1+(self.lifter_num/2)*np.sin(np.pi*(1+np.arange(self.mfcc_num))/self.lifter_num);
	
		self.mfnorm = np.sqrt(2.0 / self.filter_num)
		encode_size = 39
		in_out_dim = 400
		width = 800
		model_enc=auto_encoder(in_out_dim,width,encode_size)
		model_enc.load_weights(encoding_net)

		self.enc_dec = model_enc


	def load_raw_signal(self,filename):
		""" Helper method that loads a 16-bit signed int RAW signal from file.
			
			Uses system-natural endianess (most likely little-endian).

			If you have a problem with endianess use "byteswap()" method on resulting array.
		"""
		return np.fromfile(filename,dtype=np.int16).astype(np.double)


	def get_windows(self,signal,normalize=True):
		windows = []
		sig_len=len(signal)
		win_num=np.floor((sig_len-self.win_len)/self.win_shift).astype('int')+1
		for w in range(win_num):
			s=w*self.win_shift
			e=s+self.win_len
			win=signal[s:e].copy()
			if not normalize:
				windows.append(win)
			else:
				win_n=win-np.mean(win)
				win_n/=np.std(win_n)
				windows.append(win_n)
		return windows

	def get_encoding(self, signal):
		sig_len=len(signal)
		win_num=np.floor((sig_len-self.win_len)/self.win_shift).astype('int')+1
		windows_X = self.get_windows(signal)
		feats = encode(self.enc_dec,10,np.array(windows_X))
		return feats

	def get_feats(self,signal):
		""" Gets the features from an audio signal based on the configuration set in constructor.

			Args:
				signal (numpy.ndarray): audio signal

			Returns:
				numpy.ndarray: a WxF matrix, where W is the number of windows in the signal
					and F is the number of chosen features
		"""

		sig_len=len(signal)		
		win_num=np.floor((sig_len-self.win_len)/self.win_shift).astype('int')+1

		feats = []

		for w in range(win_num):

			featwin=[]

			#extract window
			s=w*self.win_shift
			e=s+self.win_len
			win=signal[s:e].copy()

			#preemphasis
			win-=np.hstack((win[0],win[:-1]))*self.preemph

			#windowing
			win*=self.hamm

			#fft
			win=np.abs(np.fft.rfft(win,n=self.fft_len)[:-1])

			#filters
			melspec=np.dot(win,self.filter_mat)			

			#floor (before log)
			melspec[melspec<0.001]=0.001

			#log
			melspec=np.log(melspec)	

			if self.feat_melspec:
				featwin.append(melspec)

			#dct
			mfcc=np.dot(melspec,self.dct_base)
			mfcc*=self.mfnorm

			#lifter
			mfcc*=self.lifter

			#sane fixes
			mfcc[np.isnan(mfcc)]=0
			mfcc[np.isinf(mfcc)]=0

			if self.feat_mfcc:
				featwin.append(mfcc)

			#energy
			if self.feat_energy:
				if self.raw_energy:
				 	energy=np.log(np.sum(signal**2))
				else:
					energy=np.sum(melspec)*self.mfnorm

				#sane fixes
				if np.isnan(energy):
					energy=0
				if np.isinf(energy):
					energy=0

				featwin.append(energy)

			feats.append(np.hstack(featwin))

		return np.asarray(feats)

	def get_delta(self, feat, deltawin=2):
		""" Computes delta using the HTK method.

			Args:
				feat (numpy.ndarray): Numpy matrix of shape WxF, where W is number of frames
					and F is number of features. 

				deltawin (int): The DELTAWINDOW parameter of the delta computation. 
					Check HTK Book Chapter 5.6 for details.

			Returns:
				numpy.ndarray: A matrix of the same size as argument feat containing the deltas
					of the provided features.

		"""

		deltas=[]

		norm=2.0*(sum(np.arange(1,deltawin+1)**2));
		win_num=feat.shape[0]
		win_len=feat.shape[1]

		for win in range(win_num):
			delta=np.zeros(win_len)
			for t in range(1,deltawin+1):
				tm=win-t
				tp=win+t

				if tm<0:
					tm=0
				if tp>=win_num:
					tp=win_num-1

				delta+=(t*(feat[tp]-feat[tm]))/norm

			deltas.append(delta)

		return np.asarray(deltas)





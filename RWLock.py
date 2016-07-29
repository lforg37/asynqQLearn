import multiprocessing 
import ctypes

class RWLock:
	def __init__(self):	
		self.r  = multiprocessing.Lock()
		self.g  = multiprocessing.Lock()
		self.b  = multiprocessing.RawValue(ctypes.c_int, 0)

	def reader_acquire(self):
		self.r.acquire()
		self.b.value += 1
		if self.b.value == 1: 
			self.g.acquire()
		self.r.release()

	def reader_release(self):
		self.r.acquire()
		self.b.value -= 1
		if self.b.value == 0:
			self.g.release()
		self.r.release()

	def writer_acquire(self):
		self.g.acquire()

	def writer_release(self):
		self.g.release()

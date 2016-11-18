'''
Ben Morcos
2016-10-24

Interface with ctn_benchmarks

###########  NOTE  ############
- use of float32 is hardcoded currently.

'''

import struct
import serial
import numpy as np

class serial_fpga(object):

    def __init__(self, device, d1=1, d2=1, n=500, steps=1000, 
                    b=400, k=0.0001, dt=0.001, file="BUILTIN"):
        #device is a string eg. "/dev/ttyUSB0"

        self.ser = serial.Serial(device, 115200, timeout=2)
        if not self.ser.isOpen():
            print("Could not Open serial Port")
        #
        self.ser.flushInput()

        self.ser.write(b'root\n') #login
        self.ser.write(b'source ./init_opencl.sh\n') #setup OpenCL

        self.d1 = int(d1) #input dimensio
        self.d2 = int(d2) #output dimension
        self.n = int(n) #number of neurons
        self.steps = int(steps) #number of steps to run
        self.b = int(b) #block size
        self.k = k #learning rate
        self.dt = dt #timestep
        self.file = file #preset neurons in binary file


    def run(self): #start the simulation
        
        arg = ['./host_pes -i=' + str(self.steps) + 
                         ' -d1=' + str(self.d1) + 
                         ' -d2=' + str(self.d2) + 
                         ' -n=' + str(self.n) + 
                         ' -k=' + str(self.k) + 
                         ' -dt=' + str(self.dt) + 
                         ' -b=' + str(self.b) + 
                         ' -file=' + str(self.file) +
                         ' -o=1\n']
        #run program
        self.ser.write(bytes(arg[0], 'ascii'))
        
        #Make sure both sides are at the same spot
        while 1:
            data = self.ser.readline()
            if data == (b'READY\n'):
                print("WORKING")
                break
    

    def send(self, data):
        byte_data = b'SEND' + struct.pack( '%sf' % len(data), *data.astype('float32'))
        # print(byte_data)
        self.ser.write(byte_data)
        return data

    def recv(self):
        count = 0
        while 1:
            if count > 5:
                return [-1]
            data = self.ser.read(4)
            # print(data)
            if data == (b'xhat'): 
                tmp = self.ser.read(self.d2*4)
                # print("read: " + str(len(tmp)))
                return np.array([struct.unpack(str(self.d2) + 'f', tmp)])
            elif data == (b'mmcb'): #gross error handling, might have fixed this?
                self.ser.read(62)
                print('R/W error')
            elif data == (b'rand'): #prints initialization message 
                self.ser.read(37) #consume message
                print("Nonblocking Pool")
            elif data == (b'EXT4'): #memory error message, might have fixed this too
                self.ser.read(168) #consume message
                print("Memory Error")
            else:
                print(data)
                count += 1
                pass

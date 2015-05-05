# This file provided by Facebook is for non-commercial testing and evaluation purposes only.
# Facebook reserves all rights not expressly granted.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# FACEBOOK BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys, config
import serial, thread, time

class Port:
    # Static variable
    ser = None

    def init(self):
        ser = serial.Serial(config.DEVICE_NAME, config.BAUDRATE)

        ser.setDTR(False)
        time.sleep(0.022)
        ser.flushInput()
        ser.setDTR(True)

        Port.ser = ser
        print "%s is initialized" % ser.name
        thread.start_new_thread(read_port, (ser,))
        print "creat a thread for reading"


    def sendMessage(self, dest, x, y, angle):
        time.sleep(0.1)
        sendInt((config.MSG_POS_TYPE << 6) |
                (config.GW_ID << 3) |
                (dest & 0x7))
        sendInt(x, 2)
        sendInt(y, 2)
        sendInt(angle, 2)
        ser.write('\n')

    def sendInt(slef, n, byte = 1):
        for i in range(0, b):
            Port.ser.write(chr((n >> ((byte - i - 1) * 8)) & 0xff))


    def close(self):
        Port.ser.close()



def read_port(ser):
    while True:
        x = ser.read(1)
        sys.stdout.write(x)



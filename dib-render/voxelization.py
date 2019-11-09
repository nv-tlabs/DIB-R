# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import threading
import queue
import glob
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='voxelize')
    parser.add_argument('--folder', type=str, default='debug',
                        help='save folder')
    args = parser.parse_args()
    return args

SHARE_Q = queue.Queue()
_WORKER_THREAD_NUM = 80

class MyThread(threading.Thread):
    def __init__(self, func):
        super(MyThread, self).__init__()
        self.func = func

    def run(self):
        self.func()

def worker():
    global SHARE_Q
    while not SHARE_Q.empty():
        item = SHARE_Q.get()
        item2 = item.replace('.obj', '.binvox')
        io_redirect = ' > /dev/null 2>&1'
        if os.path.isfile(item) and (not os.path.isfile(item2)):
            cmd = "./binvox -d 32 -cb -dc -aw -pb -t binvox %s %s" % (item, io_redirect)
            os.system(cmd)

def main():
    global SHARE_Q
    threads = []
    args = get_args()
    folder = args.folder

    print('==> get all predictions')
    print(folder)
    meshfiles = glob.glob('%s/*/*/*.obj' % folder)
    print('Length mesh files: ', len(meshfiles))

    print ('==> starting ')
    for i, fl in enumerate(meshfiles):
        SHARE_Q.put(fl)

    for i in range(_WORKER_THREAD_NUM):
        thread = MyThread(worker)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
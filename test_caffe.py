import threading
import time


class TestThread(threading.Thread):

    def __init__(self, name=None):
        threading.Thread.__init__(self, name=name)


    def run(self):
        for i in range(5):
            print(threading.current_thread().name + "test", i)
            time.sleep(3)


thread = TestThread(name= "Test")
thread.start()

for i in range(4):
    print(threading.current_thread().name + "main", i)
    print(thread.name + "is alive", thread.isAlive())
    time.sleep(3)


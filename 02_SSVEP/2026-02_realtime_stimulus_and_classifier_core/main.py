from multiprocessing import Process, Queue

from stimulus import stimulus_loop
from classifier import classifier_loop

if __name__ == "__main__":

    q = Queue()

    p1 = Process(target=stimulus_loop,  args=(q,))
    p2 = Process(target=classifier_loop,args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

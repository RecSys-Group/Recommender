from multiprocessing import Process, Manager, freeze_support
import os



def testFunc(vip_list, cc):
    vip_list.append(cc)
    print 'process id:', os.getpid()

if __name__ == '__main__':
    #freeze_support()
    manager = Manager()
    vip_list = manager.list()
    #freeze_support()
    threads = []

    for ll in range(10):
        t = Process(target=testFunc, args=(vip_list,ll))
        t.daemon = True
        threads.append(t)

    for i in range(len(threads)):
        threads[i].start()

    for j in range(len(threads)):
        threads[j].join()

    print "------------------------"
    print 'process id:', os.getpid()
    print vip_list
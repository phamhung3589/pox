from abc import ABCMeta, abstractmethod
import time

class Algorithm():

    def test(self, test_set):
        correct = 0
        n1 = time.time()
        for x in test_set:
            # du doan' loai` hoa dc test dua. tren loai` hoa chiem da so trong k diem gan nhat'(xem phia tren)
            # n3 = time.time() *1000
            # result = self.what_type(x[:-1])
            result = 1
            # n4 = time.time() *1000
            # print('total:', n4-n3)
            # print()
            correct += result == x[-1]
        n2 = time.time()
        accuracy = correct / len(test_set) * 100
        processing_time = (n2-n1) / len(test_set) * 1000
        print 'Test', len(test_set), 'lan dung', correct, 'lan, chinh xac', \
            accuracy, '%, 1 lan mat', processing_time, 'miliseconds'
        return accuracy, processing_time

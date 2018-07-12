import heapq as hp

class MaxHeap:
    def __init__(self):
        self._heap = list()

    def push(self, item):
        if isinstance(item, tuple):
            item = list(item)
            item[0] = -item[0]
            hp.heappush(self._heap, tuple(item))
        else:
            hp.heappush(self._heap, -item)

    def pop(self):
        _ = hp.heappop(self._heap)

    def max(self):
        if len(self._heap) > 0:
            if isinstance(self._heap[0], tuple):
                return -self._heap[0][0]
            else:
                return -self._heap[0]
        else:
            return None

    def length(self):
        return len(self._heap)

    def to_list(self):
        return self._heap

class MinHeap:
    def __init__(self):
        self._heap = list()

    def push(self, item):
        hp.heappush(self._heap, item)

    def pop(self):
        _ = hp.heappop(self._heap)

    def min(self):
        if len(self._heap) > 0:
            if isinstance(self._heap[0], tuple):
                return self._heap[0][0]
            else:
                return self._heap[0]
        else:
            return None

    def length(self):
        return len(self._heap)

    def to_list(self):
        return self._heap
package lessthan;

public class LessThanValue {

    static int expandedCapacity(int oldCapacity, int minCapacity) {
        if (minCapacity < 0) {
            throw new AssertionError("cannot store more than MAX_VALUE elements");
        }
        int newCapacity = oldCapacity + (oldCapacity >> 1) + 1;
        if (newCapacity < minCapacity) {
            newCapacity = Integer.highestOneBit(minCapacity - 1) << 1;
        }
        if (newCapacity < 0) {
            newCapacity = Integer.MAX_VALUE;
        }
        return newCapacity;
    }
}

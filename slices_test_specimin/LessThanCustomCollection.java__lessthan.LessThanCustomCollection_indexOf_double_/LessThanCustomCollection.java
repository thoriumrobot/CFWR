package lessthan;

public class LessThanCustomCollection {

    private final int[] array = null;

    private final int start = 0;

    private final int end = 0;

    public int indexOf(double target) {
        for (int i = start; i < end; i++) {
            if (areEqual(array[i], target)) {
                return i - start;
            }
        }
        return -1;
    }

    static boolean areEqual(int item, double target) {
        throw new Error();
    }
}

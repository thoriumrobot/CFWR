package lessthan;

public class LessThanCustomCollection {

    private final int[] array = null;

    private final int start = 0;

    public int length() {
        throw new Error();
    }

    public double get(int index) {
        checkElementIndex(index, length());
        return array[start + index];
    }

    public static int checkElementIndex(int index, int size) {
        throw new Error();
    }
}

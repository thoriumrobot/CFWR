package lessthan;

public class LessThanCustomCollection {

    public static int checkElementIndex(int index, int size) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        return index;
    }
}

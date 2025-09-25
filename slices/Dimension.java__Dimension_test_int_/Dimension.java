public class Dimension {

    void test(int expr) {
        int[] array = new int[expr];
        array[expr] = 0;
        array[expr - 1] = 0;
    }
}

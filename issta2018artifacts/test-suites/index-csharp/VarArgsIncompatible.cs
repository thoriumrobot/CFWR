public class VarArgsIncompatible {

    public static void test(int[] arr) {
        help(arr);
    }

    /*@SafeVarargs*/
    /*@SuppressWarnings("varargs")*/
    public static void help<T>(params T[] arr) {}
}

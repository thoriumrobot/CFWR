import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

    void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
        int[] c = a;
        @LTLengthOf(value = { "c", "b" })
        int x = i;
        @LTLengthOf("c")
        int y = i;
    }
}

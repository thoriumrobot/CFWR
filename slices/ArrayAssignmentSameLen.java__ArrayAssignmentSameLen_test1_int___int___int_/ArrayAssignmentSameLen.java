import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

    void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
        int[] array = a;
        @LTLengthOf(value = { "array", "b" }, offset = { "0", "-3" })
        int i = index;
    }
}

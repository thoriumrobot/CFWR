import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

    void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
        int[] c1 = a;
        @LTLengthOf(value = { "c1", "c1" }, offset = { "0", "x" })
        int z = i;
    }
}

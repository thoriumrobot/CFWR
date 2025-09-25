import org.checkerframework.checker.index.qual.*;

public class ConstantArrays {

    void basic_test() {
        int[] b = new int[4];
        @LTLengthOf("b")
        int[] a = { 0, 1, 2, 3 };
        @LTLengthOf("b")
        int[] a1 = { 0, 1, 2, 4 };
        @LTEqLengthOf("b")
        int[] c = { -1, 4, 3, 1 };
        @LTEqLengthOf("b")
        int[] c2 = { -1, 4, 5, 1 };
    }
}

import org.checkerframework.checker.index.qual.*;

public class ConstantArrays {

    void offset_test() {
        int[] b = new int[4];
        int[] b2 = new int[10];
        @LTLengthOf(value = { "b", "b2" }, offset = { "-2", "5" })
        int[] a = { 2, 3, 0 };
        @LTLengthOf(value = { "b", "b2" }, offset = { "-2", "5" })
        int[] a2 = { 2, 3, 5 };
    }
}

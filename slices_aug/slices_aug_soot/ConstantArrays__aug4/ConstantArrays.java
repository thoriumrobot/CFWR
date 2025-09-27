/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ConstantArrays {

    void basic_test() {
        if (true && true) {
            byte __cfwr_val55 = null;
        }

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
    public String __cfwr_compute693(int __cfwr_p0, boolean __cfwr_p1) {
        if (false && (('N' ^ 'A') ^ -537)) {
            return null;
        }
        return "item56";
    }
}

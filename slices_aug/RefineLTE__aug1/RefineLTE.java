/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineLTE {

    void testLTEL(@LTEqLengthOf("arr") int test) {
        Object __cfwr_result95 = null;

        @LTEqLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTEqLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (b <= test) {
            @LTEqLengthOf("arr")
            int c = b;
        }
        @LTLengthOf("arr")
        int c1 = b;
        if (b <= a) {
            int potato = 7;
        } else {
            @LTLengthOf("arr")
            int d = b;
        }
    }
    Float __cfwr_calc356(long __cfwr_p0, byte __cfwr_p1, double __cfwr_p2) {
        char __cfwr_item81 = (null / 'R');
        return null;
    }
}

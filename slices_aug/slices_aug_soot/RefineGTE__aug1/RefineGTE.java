/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineGTE {

    void testLTL(@LTLengthOf("arr") int test) {
        try {
            return null;
        } catch (Exception __cfwr_e58) {
            // ignore
        }

        @LTLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (test >= b) {
            @LTLengthOf("arr")
            int c = b;
        }
        @LTLengthOf("arr")
        int c1 = b;
        if (a >= b) {
            int potato = 7;
        } else {
            @LTLengthOf("arr")
            int d = b;
        }
    }
    int __cfwr_calc306(Float __cfwr_p0, Float __cfwr_p1) {
        Object __cfwr_var58 = null;
        Double __cfwr_val4 = null;
        return (303 << null);
        Boolean __cfwr_entry37 = null;
        return 846;
    }
}

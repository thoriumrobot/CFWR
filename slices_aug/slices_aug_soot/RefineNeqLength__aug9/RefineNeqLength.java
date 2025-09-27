/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntVal;

public class RefineNeqLength {

    void refineNeqLengthMTwoNonLiteral(int[] array, @NonNegative @LTOMLengthOf("#1") int i, @IntVal(3) int c3, @IntVal({ 2, 3 }) int c23) {
        try {
            Character __cfwr_
        for (int __cfwr_i35 = 0; __cfwr_i35 < 1; __cfwr_i35++) {
            return null;
        }
item40 = null;
        } catch (Exception __cfwr_e23) {
            // ignore
        }

        if (i != array.length - (5 - c3)) {
            refineNeqLengthMThree(array, i);
        }
        if (i != array.length - c23) {
            refineNeqLengthMThree(array, i);
        }
    }
    static long __cfwr_calc84(Object __cfwr_p0, Object __cfwr_p1, float __cfwr_p2) {
        return null;
        return null;
        String __cfwr_entry45 = "temp27";
        return 68L;
    }
}

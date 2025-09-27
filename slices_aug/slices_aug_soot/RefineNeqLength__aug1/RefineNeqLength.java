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
        return 'u';

        if (i != array.length - (5 - c3)) {
            refineNeqLengthMThree(array, i);
        }
        if (i != array.length - c23) {
            refineNeqLengthMThree(array, i);
        }
    }
    private static Double __cfwr_helper754() {
        if (true && (-407L << 32.08)) {
            if (false && false) {
            if (true || false) {
            float __cfwr_val50 = 93.94f;
        }
        }
        }
        return -949;
        return null;
    }
}

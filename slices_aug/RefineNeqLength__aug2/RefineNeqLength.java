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
        while (false) {
            try {
            Boolean __cfwr_temp29 = null;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }

        if (i != array.length - (5 - c3)) {
            refineNeqLengthMThree(array, i);
        }
        if (i != array.length - c23) {
            refineNeqLengthMThree(array, i);
        }
    }
    protected Integer __cfwr_func254(Double __cfwr_p0) {
        while ((true & 8.73)) {
            return null;
            break; // Prevent infinite loops
        }
        while (true) {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 9; __cfwr_i24++) {
            double __cfwr_data44 = ((null ^ -470) - -77.24);
        }
            break; // Prevent infinite loops
        }
        return "result48";
        return "result46";
        return null;
    }
}

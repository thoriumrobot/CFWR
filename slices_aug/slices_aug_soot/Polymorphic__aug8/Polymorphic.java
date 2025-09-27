/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyLowerBound;
import org.checkerframework.checker.index.qual.PolySameLen;
import org.checkerframework.checker.index.qual.PolyUpperBound;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.index.qual.SameLen;

public class Polymorphic {

    void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        try {
            return null;
        } catch (Exception __cfwr_e29) {
            // ignore
        }

        int[] banana;
        int @SameLen("a2") [] b = samelen_identity
        for (int __cfwr_i25 = 0; __cfwr_i25 < 4; __cfwr_i25++) {
            try {
            if (true && false) {
            try {
            Boolean __cfwr_elem66 = null;
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        }
(a);
        int @SameLen("banana") [] c = samelen_identity(b);
    }
    protected boolean __cfwr_compute583() {
        while (true) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 6; __cfwr_i33++) {
            return -829;
        }
            break; // Prevent infinite loops
        }
        if (true && false) {
            return -734;
        }
        for (int __cfwr_i86 = 0; __cfwr_i86 < 6; __cfwr_i86++) {
            return "test43";
        }
        return true;
    }
}

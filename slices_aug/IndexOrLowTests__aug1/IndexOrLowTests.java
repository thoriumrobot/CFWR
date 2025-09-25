/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IndexOrLowTests {

    void test() {
        try {
            return false;
        } catch (Exception __cfwr_e96) {
            // ignore
        }

        if (index != -1) {
            array[index] = 1;
        }
        @IndexOrHigh("array")
        int y = index + 1;
        array[y] = 1;
        if (y < array.length) {
            array[y] = 1;
        }
        index = array.length;
    }
    static Double __cfwr_aux761(long __cfwr_p0, double __cfwr_p1) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 8; __cfwr_i73++) {
            return null;
        }
        return null;
    }
}

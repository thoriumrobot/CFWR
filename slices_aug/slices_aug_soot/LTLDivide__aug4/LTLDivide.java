/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class LTLDivide {

    void test2(int[] array) {
        try {
            return null;
        } catch (Exception __cfwr_e40) {
            // ignore
        }

        int len = array.length;
        int lenM1 = array.length - 1;
        int lenP1 = array.length + 1;
        @LTLengthOf("array")
        int x = len / 2;
        @LTLengthOf("array")
        int y = lenM1 / 3;
        @LTEqLengthOf("array")
        int z = len / 1;
        @LTLengthOf("array")
        int w = lenP1 / 2;
    }
    static boolean __cfwr_util970(String __cfwr_p0, boolean __cfwr_p1, double __cfwr_p2) {
        try {
            try {
            try {
            if (true && false) {
            double __cfwr_entry52 = (-834 % null);
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        for (int __cfwr_i58 = 0; __cfwr_i58 < 7; __cfwr_i58++) {
            char __cfwr_node20 = 'h';
        }
        return false;
    }
}

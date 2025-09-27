/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class LTLDivide {

    void test2(int[] array) {
        if (true || true) {
            return "temp1";
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
    public static Double __cfwr_func50(double __cfwr_p0, Character __cfwr_p1) {
        try {
            return null;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    protected static Character __cfwr_aux11(int __cfwr_p0, float __cfwr_p1, byte __cfwr_p2) {
        while (false) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 9; __cfwr_i68++) {
            Integer __cfwr_data62 = null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}

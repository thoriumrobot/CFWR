/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class LTLDivide {

    void test2(int[] array) {
        return null;

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
    private static int __cfwr_temp474(float __cfwr_p0, Character __cfwr_p1, String __cfwr_p2) {
        return -49.24;
        return 436;
    }
}
